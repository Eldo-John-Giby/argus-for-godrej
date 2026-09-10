"""VLM verification layer — semantic confirmation + explanation of candidate events.

Uses Qwen3-VL (primary) or Gemini 3 Flash (fallback) to verify candidate
events from the FSM. Only called on candidate clips, never per-frame.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    QWEN_API_KEY,
    VLM_PROVIDER,
    VLM_TIMEOUT_SECONDS,
)
from app.vlm.explain import generate_explanation


@dataclass
class VLMResult:
    """Output from VLM verification."""
    verified: bool  # Does the VLM agree this is a real event?
    explanation: str  # Human-readable explanation
    confidence: float  # VLM's own confidence (0–1)
    raw_response: Optional[str] = None


VERIFICATION_PROMPT = """You are an AI safety inspector reviewing warehouse handling footage.

A computer vision system has flagged this clip as a potential safety violation.
Analyse the clip and determine:

1. Is this genuinely a safety/handling violation? (yes/no)
2. What exactly is happening? (1-2 sentences)
3. How confident are you? (0.0–1.0)

Flagged behaviour: {behaviour_type}
System features: {features_json}

Respond in JSON format:
{{
  "verified": true/false,
  "explanation": "...",
  "confidence": 0.0-1.0
}}
"""


class VLMVerifier:
    """Verify candidate events using a Vision-Language Model."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or VLM_PROVIDER

    async def verify_event(
        self,
        clip_path: Path,
        behaviour_type: str,
        features: dict,
    ) -> VLMResult:
        """Verify a candidate event clip using the VLM.

        Args:
            clip_path: Path to the extracted video clip or keyframe image.
            behaviour_type: The flagged behaviour type string.
            features: Dictionary of computed features for this event.

        Returns:
            VLMResult with verification status and explanation.
        """
        prompt = VERIFICATION_PROMPT.format(
            behaviour_type=behaviour_type,
            features_json=json.dumps(features, indent=2),
        )

        # Read clip/keyframe as base64
        image_b64 = self._encode_media(clip_path)

        if self.provider == "gemini":
            result = await self._verify_gemini(prompt, image_b64)
        else:
            result = await self._verify_qwen(prompt, image_b64)

        # When no VLM is reachable (no key, no credits, offline demo) the
        # explanation panel must still say something true and specific —
        # fall back to the deterministic, feature-grounded narrative.
        if not result.verified and (
            "not configured" in result.explanation
            or "API error" in result.explanation
            or "API exception" in result.explanation
            or result.explanation.startswith("Could not parse")
        ):
            result.explanation = generate_explanation(behaviour_type, features)
        return result

    async def _verify_qwen(self, prompt: str, image_b64: str) -> VLMResult:
        """Call Qwen3-VL API for verification."""
        if not QWEN_API_KEY:
            return VLMResult(
                verified=False,
                explanation="VLM not configured (missing QWEN_API_KEY)",
                confidence=0.0,
            )

        try:
            import httpx

            async with httpx.AsyncClient(timeout=VLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct",
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                    json={
                        "inputs": prompt,
                        "parameters": {"max_new_tokens": 512},
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    text = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
                    return self._parse_vlm_response(text)
                else:
                    return VLMResult(
                        verified=False,
                        explanation=f"Qwen API error: {response.status_code}",
                        confidence=0.0,
                    )
        except Exception as e:
            return VLMResult(
                verified=False,
                explanation=f"Qwen API exception: {str(e)}",
                confidence=0.0,
            )

    async def _verify_gemini(self, prompt: str, image_b64: str) -> VLMResult:
        """Call Gemini (Generative Language API REST) for verification.

        Uses httpx directly — no google.generativeai SDK dependency. The SDK's
        1.5-flash path is deprecated and newer models reject it.
        """
        if not GEMINI_API_KEY:
            return VLMResult(
                verified=False,
                explanation="VLM not configured (missing GEMINI_API_KEY)",
                confidence=0.0,
            )

        try:
            import httpx

            parts: list[dict] = [{"text": prompt}]
            if image_b64:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64,
                    }
                })

            async with httpx.AsyncClient(timeout=VLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                    headers={"x-goog-api-key": GEMINI_API_KEY},
                    json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {
                            "maxOutputTokens": 512,
                            "temperature": 0.1,
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"] or ""
                    except (KeyError, IndexError, TypeError):
                        text = ""
                    return self._parse_vlm_response(text)

                # Auth/config problems must surface loudly, silently swallowing
                # them would leave VLM verification dead with no hint why.
                try:
                    msg = response.json().get("error", {}).get("message", "")
                except Exception:
                    msg = response.text[:200]
                return VLMResult(
                    verified=False,
                    explanation=f"Gemini API error {response.status_code}: {msg[:200]}",
                    confidence=0.0,
                )
        except Exception as e:
            return VLMResult(
                verified=False,
                explanation=f"Gemini API exception: {str(e)}",
                confidence=0.0,
            )

    def _parse_vlm_response(self, text: str) -> VLMResult:
        """Parse VLM JSON response into VLMResult."""
        try:
            # Try to extract JSON from the response
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)
            return VLMResult(
                verified=data.get("verified", False),
                explanation=data.get("explanation", "No explanation provided"),
                confidence=float(data.get("confidence", 0.5)),
                raw_response=text,
            )
        except (json.JSONDecodeError, ValueError, IndexError):
            return VLMResult(
                verified=False,
                explanation=f"Could not parse VLM response: {text[:200]}",
                confidence=0.0,
                raw_response=text,
            )

    def _encode_media(self, path: Path) -> str:
        """Encode a media file to base64."""
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
