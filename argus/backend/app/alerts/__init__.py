"""Real-time alert delivery — multilingual voice + TTS for High/Critical events.

The frontend already polls /api/events/ every 5 s; this module only needs to
render an alert line for an event it already fetched. Chain:

  event -> alert text (EN) -> translation (GPT-based translator, optional) -> TTS (gTTS, optional)

Both steps are off-the-shelf; neither is required. Without keys/network the
alert text is still composed locally from DB fields (behaviour, bay, tier,
risk score), and the frontend falls back to the browser SpeechSynthesis API —
so the multilingual alert always works in the demo, online or offline.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Optional

import httpx

from app.config import (
    ALERT_TRANSLATION_API_KEY,
    ALERT_TRANSLATION_MODEL,
    ALERT_TRANSLATION_URL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TTS_PROVIDER,
    TTS_TIMEOUT_SECONDS,
)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "kn": "Kannada",
}

TIER_URGENCY = {"Critical": "URGENT", "High": "Attention", "Medium": "Notice", "Low": "Notice"}


def compose_alert_text(
    behaviour_type: str,
    tier: str,
    risk_score: float,
    bay_id: Optional[str] = None,
    camera_name: Optional[str] = None,
    timestamp: Optional[dt.datetime] = None,
) -> str:
    """Build the spoken alert line locally from event fields.

    Deterministic, no network, no LLM — always available as the base text
    that translation (if any) starts from.
    """
    behaviour = behaviour_type.replace("_", " ").replace("-", " ").strip()
    where = f"in {camera_name}" if camera_name else (f"in bay {bay_id}" if bay_id else "")
    urgency = TIER_URGENCY.get(tier, "Notice")
    when = f" at {timestamp.strftime('%H:%M')}" if timestamp else ""
    return f"{urgency}: {behaviour} detected {where}{when}. Risk score {risk_score:.0f} out of 100."


async def translate_text(text: str, target_lang: str) -> dict:
    """Translate the alert text to the target language.

    Uses an OpenAI-compatible chat endpoint configured via env vars
    (ALERT_TRANSLATION_URL + ALERT_TRANSLATION_API_KEY). Falls back to the
    Gemini key already used by the VLM verifier when set. If neither is
    configured or the call fails, returns the original text with
    translated=False so the caller always gets something speakable.
    """
    lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
    if target_lang == "en" or not text:
        return {"text": text, "translated": False, "language": "en"}

    prompt = (
        f"Translate this warehouse safety alert into {lang_name} "
        f"(language code {target_lang}). Keep it short, direct and natural for "
        f"a spoken announcement. Reply with the translation only.\n\n{text}"
    )

    # Preferred: generic OpenAI-compatible translator endpoint
    if ALERT_TRANSLATION_API_KEY and ALERT_TRANSLATION_URL:
        try:
            async with httpx.AsyncClient(timeout=TTS_TIMEOUT_SECONDS) as client:
                res = await client.post(
                    ALERT_TRANSLATION_URL,
                    headers={"Authorization": f"Bearer {ALERT_TRANSLATION_API_KEY}"},
                    json={
                        "model": ALERT_TRANSLATION_MODEL,
                        "temperature": 0.2,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    out = data["choices"][0]["message"]["content"].strip()
                    if out:
                        return {"text": out, "translated": True, "language": target_lang}
        except Exception:
            pass  # fall through to Gemini, then to the original text

    # Fallback: reuse the Gemini key if present (REST, no SDK dependency)
    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=TTS_TIMEOUT_SECONDS) as client:
                res = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                    headers={"x-goog-api-key": GEMINI_API_KEY},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 256, "temperature": 0.2},
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if out:
                        return {"text": out, "translated": True, "language": target_lang}
        except Exception:
            pass  # fall through to the original text — alerts must never block

    # No translation available — speak the English base text.
    return {"text": text, "translated": False, "language": "en"}


async def synthesize_speech(text: str, lang: str) -> tuple[Optional[bytes], Optional[str]]:
    """Render text to audio bytes via gTTS.

    Returns (audio_bytes, content_type). (None, None) when TTS is disabled,
    gTTS is not installed, or synthesis fails — the frontend then falls back
    to browser SpeechSynthesis.
    """
    if TTS_PROVIDER != "gtts":
        return None, None
    try:
        from gtts import gTTS  # optional dependency

        from io import BytesIO

        buf = BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        return buf.getvalue(), "audio/mpeg"
    except Exception:
        return None, None


def parse_lang(value: Optional[str]) -> str:
    """Accept 'hi', 'hi-IN', 'Hindi' etc. and normalise to a supported code."""
    if not value:
        return "en"
    v = value.strip().lower()
    code = v.split("-")[0]
    for known in SUPPORTED_LANGUAGES:
        if v == known or code == known or v == SUPPORTED_LANGUAGES[known].lower():
            return known
    return "en"


def render_sse(event: dict) -> str:
    """Format a dict as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"
