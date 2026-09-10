"""Risk scoring engine — physics-informed, config-driven.

Formula:
  risk_score = w1*severity + w2*impact + w3*fragility + w7*location
             + w4*stack_instability + w5*recurrence - w6*confidence_penalty

All weights and thresholds are config-driven (see config.py RISK_WEIGHTS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import (
    RISK_TIER_HIGH,
    RISK_TIER_LOW,
    RISK_TIER_MEDIUM,
    RISK_WEIGHTS,
    bay_hazard_factor,
)


def tier_for_score(score: float) -> str:
    """Map a 0-100 risk score to its tier using the configured bands.

    Single source of truth for tier assignment — used by score_event and by
    the DB seeder so stored reports never drift from the active calibration.
    """
    if score >= RISK_TIER_HIGH:
        return "Critical"
    if score >= RISK_TIER_MEDIUM:
        return "High"
    if score >= RISK_TIER_LOW:
        return "Medium"
    return "Low"


@dataclass
class RiskInput:
    """Inputs for computing a single event's risk score."""
    behaviour_severity: float  # base severity from taxonomy (0-100)
    velocity_at_contact: float  # px/s, calibrated
    mass_proxy: float  # relative mass (1.0 = standard carton)
    fragility_multiplier: float  # 0.7 (rugged) – 1.5 (fragile)
    stack_tilt_angle: float  # degrees
    top_bottom_footprint_ratio: float
    recurrence_count: int  # same behaviour+bay+operator in rolling window
    composite_confidence: float  # 0.0 – 1.0
    bay_id: str | None = None  # location input — per-bay hazard factor (PS)


@dataclass
class RiskResult:
    """Output of risk scoring."""
    score: float  # 0 – 100
    tier: str  # Low / Medium / High / Critical
    breakdown: dict  # individual component contributions


def compute_impact_proxy(velocity: float, mass_proxy: float, height_proxy: float = 1.0) -> float:
    """Physics proxy for impact energy: 0.5 * mass * velocity² * height_factor.

    Normalized to 0–100 scale.
    """
    raw = 0.5 * mass_proxy * (velocity ** 2) * height_proxy
    # Normalize: assume max realistic impact is ~500 px/s with mass 2.0 and height 2.0
    max_impact = 0.5 * 2.0 * (500 ** 2) * 2.0
    normalized = min(raw / max_impact * 100, 100.0)
    return normalized


def compute_stack_instability(tilt_angle: float, footprint_ratio: float) -> float:
    """Compute stack instability score from tilt and footprint comparison.

    Returns 0–100.
    """
    # Tilt contribution: 0° = 0, 30°+ = max
    tilt_score = min(tilt_angle / 30.0, 1.0) * 50

    # Footprint ratio: 1.0 = neutral, 1.5+ = max instability
    fp_score = min(max(footprint_ratio - 1.0, 0) / 0.5, 1.0) * 50

    return tilt_score + fp_score


def compute_recurrence_factor(count: int) -> float:
    """Recurrence factor: repeated offences increase severity.

    Logarithmic scaling to prevent runaway scores.
    """
    if count <= 1:
        return 0.0
    return min(math.log2(count) * 15, 50.0)


def compute_confidence_penalty(confidence: float) -> float:
    """Low confidence pulls score DOWN — never inflate with uncertainty."""
    # confidence 1.0 → penalty 0; confidence 0.0 → penalty 40
    return (1.0 - confidence) * 40.0


def score_event(risk_input: RiskInput) -> RiskResult:
    """Compute risk score for a candidate event.

    Returns RiskResult with score, tier, and per-component breakdown.
    """
    w = RISK_WEIGHTS

    impact = compute_impact_proxy(
        risk_input.velocity_at_contact,
        risk_input.mass_proxy,
    )

    stack = compute_stack_instability(
        risk_input.stack_tilt_angle,
        risk_input.top_bottom_footprint_ratio,
    )

    recurrence = compute_recurrence_factor(risk_input.recurrence_count)

    confidence_penalty = compute_confidence_penalty(risk_input.composite_confidence)

    # Location factor (PS): per-bay hazard multiplier scales the location
    # component the same way fragility scales its own component.
    location_factor = bay_hazard_factor(risk_input.bay_id)

    breakdown = {
        "behaviour_severity": risk_input.behaviour_severity,
        "impact_proxy": impact,
        "fragility": risk_input.fragility_multiplier * 30,  # scaled
        "location": location_factor * 30,  # scaled, same convention as fragility
        "stack_instability": stack,
        "recurrence": recurrence,
        "confidence_penalty": -confidence_penalty,
    }

    raw_score = (
        w["w1_behaviour_severity"] * risk_input.behaviour_severity
        + w["w2_impact_proxy"] * impact
        + w["w3_fragility"] * risk_input.fragility_multiplier * 30
        + w["w7_location"] * location_factor * 30
        + w["w4_stack_instability"] * stack
        + w["w5_recurrence"] * recurrence
        - w["w6_confidence_penalty"] * confidence_penalty
    )

    # Clamp to 0–100
    score = max(0.0, min(raw_score, 100.0))

    return RiskResult(score=score, tier=tier_for_score(score), breakdown=breakdown)
