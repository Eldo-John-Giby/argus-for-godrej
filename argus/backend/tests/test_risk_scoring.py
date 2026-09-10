"""Unit tests for the risk scoring engine."""

import pytest
from app.risk.scoring import (
    RiskInput,
    RiskResult,
    compute_impact_proxy,
    compute_stack_instability,
    compute_recurrence_factor,
    compute_confidence_penalty,
    score_event,
)
from app.risk.confidence_gate import (
    ConfidenceInput,
    compute_composite_confidence,
    apply_confidence_gate,
    update_status_after_feedback,
)


class TestImpactProxy:
    """Test impact energy proxy computation."""

    def test_zero_velocity(self):
        assert compute_impact_proxy(0, 1.0) == 0.0

    def test_high_velocity(self):
        result = compute_impact_proxy(500, 2.0)
        assert result >= 50

    def test_low_velocity(self):
        result = compute_impact_proxy(10, 1.0)
        assert result < 5

    def test_mass_scales_linearly(self):
        low_mass = compute_impact_proxy(100, 0.5)
        high_mass = compute_impact_proxy(100, 2.0)
        assert high_mass > low_mass


class TestStackInstability:
    """Test stack instability computation."""

    def test_no_tilt(self):
        result = compute_stack_instability(0, 1.0)
        assert result == 0.0

    def test_high_tilt(self):
        result = compute_stack_instability(30, 1.0)
        assert result >= 50

    def test_footprint_ratio(self):
        result = compute_stack_instability(0, 1.5)
        assert result > 0

    def test_combined(self):
        result = compute_stack_instability(15, 1.2)
        assert result > 0


class TestRecurrenceFactor:
    """Test recurrence factor computation."""

    def test_no_recurrence(self):
        assert compute_recurrence_factor(1) == 0.0

    def test_single_recurrence(self):
        assert compute_recurrence_factor(2) > 0

    def test_logarithmic_scaling(self):
        r4 = compute_recurrence_factor(4)
        r8 = compute_recurrence_factor(8)
        # Should increase but sub-linearly
        assert r8 > r4
        assert r8 < 2 * r4


class TestConfidencePenalty:
    """Test confidence penalty computation."""

    def test_full_confidence(self):
        assert compute_confidence_penalty(1.0) == 0.0

    def test_zero_confidence(self):
        penalty = compute_confidence_penalty(0.0)
        assert penalty == 40.0

    def test_half_confidence(self):
        penalty = compute_confidence_penalty(0.5)
        assert 15 < penalty < 25


class TestScoreEvent:
    """Test end-to-end risk scoring."""

    def test_low_risk_event(self):
        risk_input = RiskInput(
            behaviour_severity=20,
            velocity_at_contact=10,
            mass_proxy=1.0,
            fragility_multiplier=1.0,
            stack_tilt_angle=0,
            top_bottom_footprint_ratio=1.0,
            recurrence_count=1,
            composite_confidence=0.9,
        )
        result = score_event(risk_input)
        assert result.score < 25
        assert result.tier == "Low"

    def test_high_risk_event(self):
        risk_input = RiskInput(
            behaviour_severity=90,
            velocity_at_contact=400,
            mass_proxy=2.0,
            fragility_multiplier=1.5,
            stack_tilt_angle=25,
            top_bottom_footprint_ratio=1.4,
            recurrence_count=5,
            composite_confidence=0.3,
        )
        result = score_event(risk_input)
        # Pin the SCORE, not the tier: tier bands are per-site config
        # (RISK_TIER_* in config.py) and are calibrated per deployment.
        assert result.score > 50
        assert result.tier in ("Medium", "High", "Critical")

    def test_score_clamped_to_100(self):
        risk_input = RiskInput(
            behaviour_severity=100,
            velocity_at_contact=1000,
            mass_proxy=5.0,
            fragility_multiplier=2.0,
            stack_tilt_angle=45,
            top_bottom_footprint_ratio=2.0,
            recurrence_count=20,
            composite_confidence=0.1,
        )
        result = score_event(risk_input)
        assert result.score <= 100

    def test_breakdown_provided(self):
        risk_input = RiskInput(
            behaviour_severity=50,
            velocity_at_contact=100,
            mass_proxy=1.0,
            fragility_multiplier=1.0,
            stack_tilt_angle=5,
            top_bottom_footprint_ratio=1.0,
            recurrence_count=1,
            composite_confidence=0.7,
        )
        result = score_event(risk_input)
        assert "behaviour_severity" in result.breakdown
        assert "impact_proxy" in result.breakdown
        assert "confidence_penalty" in result.breakdown


class TestConfidenceGate:
    """Test the Observed → Potential → Confirmed confidence ladder."""

    def test_low_confidence_observed(self):
        ci = ConfidenceInput(
            detector_confidence=0.3,
            track_stability=0.3,
            vlm_agreement=0.3,
        )
        result = apply_confidence_gate(ci)
        assert result.status == "observed"
        assert result.should_alert is False

    def test_medium_confidence_potential(self):
        ci = ConfidenceInput(
            detector_confidence=0.7,
            track_stability=0.6,
            vlm_agreement=0.6,
        )
        result = apply_confidence_gate(ci)
        assert result.status == "potential"
        assert result.should_alert is True
        assert result.needs_human_review is True

    def test_high_confidence_confirmed(self):
        ci = ConfidenceInput(
            detector_confidence=0.9,
            track_stability=0.9,
            vlm_agreement=0.9,
        )
        result = apply_confidence_gate(ci)
        assert result.status == "confirmed"
        assert result.should_alert is True

    def test_no_vlm_shifts_weight(self):
        ci = ConfidenceInput(
            detector_confidence=0.8,
            track_stability=0.7,
            vlm_agreement=-1,  # not called
        )
        result = apply_confidence_gate(ci)
        # Should still compute a reasonable confidence
        assert 0.5 <= result.composite_confidence <= 1.0


class TestStatusUpdate:
    """Test status updates after human feedback."""

    def test_confirm_potential_to_confirmed(self):
        assert update_status_after_feedback("potential", "confirm", 0.7) == "confirmed"

    def test_dismiss_any_to_dismissed(self):
        assert update_status_after_feedback("potential", "dismiss", 0.7) == "dismissed"
        assert update_status_after_feedback("confirmed", "dismiss", 0.9) == "dismissed"
        assert update_status_after_feedback("observed", "dismiss", 0.3) == "dismissed"

    def test_confirm_observed_to_potential(self):
        assert update_status_after_feedback("observed", "confirm", 0.4) == "potential"
