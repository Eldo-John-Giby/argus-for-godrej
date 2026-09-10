"""Confidence gate — the Observed → Potential → Confirmed ladder.

This is one of three core differentiators (alongside the feedback loop
and the grounded assistant).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIDENCE_LOW, CONFIDENCE_MEDIUM


@dataclass
class ConfidenceInput:
    """Inputs for the confidence gate."""
    detector_confidence: float  # YOLO detection confidence (0–1)
    track_stability: float  # ByteTrack ID persistence score (0–1)
    vlm_agreement: float  # VLM semantic verification agreement (0–1, -1 if not called)


@dataclass
class ConfidenceResult:
    """Output of the confidence gate."""
    composite_confidence: float
    status: str  # "observed" | "potential" | "confirmed"
    should_alert: bool
    needs_human_review: bool


def compute_composite_confidence(ci: ConfidenceInput) -> float:
    """Compute composite confidence from detector, tracker, and VLM signals.

    If VLM hasn't been called yet (vlm_agreement == -1), weight shifts
    to detector and tracker.
    """
    if ci.vlm_agreement < 0:
        # VLM not yet called — weight between detector and tracker
        return 0.6 * ci.detector_confidence + 0.4 * ci.track_stability
    else:
        # Full three-way composite
        return (
            0.35 * ci.detector_confidence
            + 0.30 * ci.track_stability
            + 0.35 * ci.vlm_agreement
        )


def apply_confidence_gate(ci: ConfidenceInput) -> ConfidenceResult:
    """Apply the confidence gate to determine event status.

    < 0.5  → Observed (log only, no alert)
    0.5–0.8 → Potential Risk (alert, queue for human review)
    > 0.8 → Confirmed (full alert + incident report, if human-confirmed at least once)
    """
    composite = compute_composite_confidence(ci)

    if composite < CONFIDENCE_LOW:
        status = "observed"
        should_alert = False
        needs_human_review = False
    elif composite < CONFIDENCE_MEDIUM:
        status = "potential"
        should_alert = True
        needs_human_review = True
    else:
        status = "confirmed"
        should_alert = True
        needs_human_review = False  # Auto-confirmed at high confidence

    return ConfidenceResult(
        composite_confidence=composite,
        status=status,
        should_alert=should_alert,
        needs_human_review=needs_human_review,
    )


def update_status_after_feedback(
    current_status: str,
    action: str,  # "confirm" | "dismiss"
    composite_confidence: float,
) -> str:
    """Update event status based on human feedback.

    - confirm on 'potential' → 'confirmed'
    - dismiss on any → 'dismissed'
    - confirm on 'observed' → 'potential' (bump up)
    """
    if action == "dismiss":
        return "dismissed"
    elif action == "confirm":
        if current_status == "observed":
            return "potential"
        elif current_status == "potential":
            return "confirmed"
        elif current_status == "confirmed":
            return "confirmed"
    return current_status
