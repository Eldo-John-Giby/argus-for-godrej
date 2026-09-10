"""Full pipeline test with realistic simulated warehouse detections.

This simulates what YOLO would produce on real warehouse footage,
then runs the full pipeline: detections → FSM → features → risk scoring
→ confidence gate → event creation.

Verifies every layer works correctly with realistic data.
"""

import json
from app.behaviour.fsm import TrackFSM, FrameData, TrackState, EventType
from app.behaviour.features import (
    extract_features, compute_tilt_angle, compute_footprint_ratio,
    compute_object_density, compute_overlap_ratio, point_in_polygon,
)
from app.risk.scoring import RiskInput, score_event, compute_impact_proxy
from app.risk.confidence_gate import ConfidenceInput, apply_confidence_gate, update_status_after_feedback


def simulate_carry_and_drop():
    """Simulate a person carrying a box, then dropping it."""
    print("=" * 60)
    print("Scenario 1: Person carrying box -> drops it")
    print("=" * 60)

    fsm = TrackFSM(track_id=1, object_class="box")
    events = []

    # Phase 1: Idle → Pickup (t=0 to t=0.5s)
    print("\n  Phase 1: Idle → Pickup")
    for i in range(15):
        t = i / 30.0
        # Person approaches, picks up box — box moves upward
        y_offset = max(0, 200 - i * 8)  # box rises from 200 to ~80
        frame = FrameData(
            timestamp=t,
            bbox=(100, y_offset, 160, y_offset + 50),
            center=(130, y_offset + 25),
            bottom_y=y_offset + 50,
            area=3000,
        )
        event = fsm.update(frame)
        if event:
            events.append(event)
            print(f"    t={t:.2f}s: EVENT — {event.event_type.value}")

    print(f"    FSM state after pickup: {fsm.state.value}")

    # Phase 2: Carrying (t=0.5s to t=2.0s) — horizontal motion
    print("\n  Phase 2: Carrying (horizontal movement)")
    for i in range(45):
        t = 0.5 + i / 30.0
        # Box moves steadily to the right at y=100
        x_offset = 130 + i * 3
        frame = FrameData(
            timestamp=t,
            bbox=(x_offset, 80, x_offset + 60, 130),
            center=(x_offset + 30, 105),
            bottom_y=130,
            area=3000,
        )
        event = fsm.update(frame)
        if event:
            events.append(event)
            print(f"    t={t:.2f}s: EVENT — {event.event_type.value}")

    print(f"    FSM state during carry: {fsm.state.value}")

    # Phase 3: DROP! (t=2.0s to t=2.3s) — sudden downward acceleration
    print("\n  Phase 3: DROP EVENT!")
    for i in range(10):
        t = 2.0 + i / 30.0
        drop_t = t - 2.0
        # Gravity: y = 0.5 * g * t² (g ≈ 800 px/s² in image coords)
        y_pos = int(130 + 0.5 * 800 * drop_t ** 2)
        y_pos = min(y_pos, 420)  # floor
        x_pos = 470  # stays roughly in place horizontally

        frame = FrameData(
            timestamp=t,
            bbox=(x_pos, y_pos, x_pos + 60, y_pos + 50),
            center=(x_pos + 30, y_pos + 25),
            bottom_y=y_pos + 50,
            area=3000,
        )
        event = fsm.update(frame)
        if event:
            events.append(event)
            print(f"    t={t:.2f}s: ⚠️  EVENT — {event.event_type.value} "
                  f"(confidence: {event.confidence:.3f})")

    print(f"\n  Total events detected: {len(events)}")
    print(f"  Final FSM state: {fsm.state.value}")

    return events


def simulate_dragging():
    """Simulate a person dragging a box along the floor."""
    print("\n" + "=" * 60)
    print("Scenario 2: Person dragging box along floor")
    print("=" * 60)

    fsm = TrackFSM(track_id=2, object_class="box")
    events = []

    # Pickup
    for i in range(5):
        t = i / 30.0
        frame = FrameData(
            timestamp=t, bbox=(50, 300, 110, 360),
            center=(80, 330), bottom_y=360, area=3600,
        )
        fsm.update(frame)

    # Carrying briefly
    for i in range(10):
        t = 0.17 + i / 30.0
        frame = FrameData(
            timestamp=t, bbox=(80, 200, 140, 260),
            center=(110, 230), bottom_y=260, area=3600,
        )
        fsm.update(frame)

    # Dragging phase: box on the floor, moving horizontally
    print("\n  Dragging phase (box along floor):")
    for i in range(90):
        t = 0.5 + i / 30.0
        x_pos = 140 + i * 4  # steady horizontal motion
        floor_y = 420  # floor level

        frame = FrameData(
            timestamp=t,
            bbox=(x_pos, floor_y - 40, x_pos + 50, floor_y),
            center=(x_pos + 25, floor_y - 20),
            bottom_y=floor_y,
            area=2000,
        )
        event = fsm.update(frame)
        if event:
            events.append(event)
            print(f"    t={t:.2f}s: ⚠️  EVENT — {event.event_type.value}")

    print(f"  Total events detected: {len(events)}")
    return events


def simulate_throwing():
    """Simulate a person throwing a box (ballistic trajectory)."""
    print("\n" + "=" * 60)
    print("Scenario 3: Person throwing box (ballistic trajectory)")
    print("=" * 60)

    fsm = TrackFSM(track_id=3, object_class="box")
    events = []

    # Build up history first (carry phase)
    for i in range(20):
        t = i / 30.0
        x = 100 + i * 5
        frame = FrameData(
            timestamp=t, bbox=(x, 200, x+50, 250),
            center=(x+25, 225), bottom_y=250, area=2500,
        )
        fsm.update(frame)

    # Throwing: parabolic trajectory (y = at² + bt + c)
    # Simulate throw from x=350, y=225 → arcs up then falls
    print("\n  Throwing phase (parabolic trajectory):")
    throw_start = 20 / 30.0
    for i in range(30):
        t = throw_start + i / 30.0
        dt = t - throw_start

        # Parabolic trajectory: starts going up, then falls
        x = 350 + i * 8  # horizontal velocity
        y = 225 - 200 * dt + 400 * dt ** 2  # parabola (up then down)

        frame = FrameData(
            timestamp=t,
            bbox=(x, int(y), x+50, int(y)+50),
            center=(x+25, int(y)+25),
            bottom_y=int(y)+50,
            area=2500,
        )
        event = fsm.update(frame)
        if event:
            events.append(event)
            print(f"    t={t:.2f}s: ⚠️  EVENT — {event.event_type.value} "
                  f"(confidence: {event.confidence:.3f})")

    print(f"  Total events detected: {len(events)}")
    return events


def test_feature_extraction():
    """Test geometry feature extraction."""
    print("\n" + "=" * 60)
    print("Feature Extraction Tests")
    print("=" * 60)

    # Tilt angle
    upright = compute_tilt_angle(30, 100)  # tall box
    tilted = compute_tilt_angle(80, 100)   # wider box
    print(f"  Upright box (30x100): tilt = {upright:.1f}°")
    print(f"  Tilted box (80x100):  tilt = {tilted:.1f}°")
    assert upright < tilted, "Tilted box should have higher tilt angle"

    # Footprint ratio (improper stacking)
    normal = compute_footprint_ratio((0,0,50,50), (0,0,80,80))   # top smaller
    improper = compute_footprint_ratio((0,0,80,80), (0,0,50,50)) # top larger
    print(f"  Normal stack footprint ratio: {normal:.2f} (should be <1)")
    print(f"  Improper stack ratio:         {improper:.2f} (should be >1)")
    assert normal < 1.0, "Normal stack: top should be smaller"
    assert improper > 1.0, "Improper stack: top is larger"

    # Zone membership
    zone = [(0,0), (200,0), (200,200), (0,200)]
    inside = point_in_polygon((100, 100), zone)
    outside = point_in_polygon((300, 100), zone)
    print(f"  Point (100,100) in zone: {inside} (should be True)")
    print(f"  Point (300,100) in zone: {outside} (should be False)")
    assert inside is True
    assert outside is False

    # Object density
    centers = [(50,50), (80,60), (120,100), (150,140), (30,90)]
    density = compute_object_density(centers, zone)
    print(f"  Object density: {density:.3f}")
    assert density > 0

    # Overlap ratio
    bboxes = [(0,0,100,100), (50,50,150,150), (200,200,300,300)]
    overlap = compute_overlap_ratio(bboxes)
    print(f"  Overlap ratio (partial): {overlap:.3f}")
    assert 0 < overlap < 1

    print("  ✅ All feature extraction tests passed")


def test_risk_scoring_pipeline(events):
    """Test risk scoring on detected events."""
    print("\n" + "=" * 60)
    print("Risk Scoring Pipeline")
    print("=" * 60)

    scored = []
    for event in events:
        # Compute features from event data
        velocity = event.features.get("downward_velocity",
                     event.features.get("horizontal_speed",
                     event.features.get("throw_velocity", (0, 0))))
        if isinstance(velocity, tuple):
            import math
            velocity_mag = math.sqrt(velocity[0]**2 + velocity[1]**2)
        else:
            velocity_mag = abs(velocity)

        risk_input = RiskInput(
            behaviour_severity={
                EventType.PRODUCT_DROPPED: 85,
                EventType.PRODUCT_DRAGGED: 55,
                EventType.PRODUCT_THROWN: 90,
                EventType.DRAGGING_NO_TROLLEY: 60,
            }.get(event.event_type, 50),
            velocity_at_contact=velocity_mag,
            mass_proxy=1.0,
            fragility_multiplier=1.0,
            stack_tilt_angle=0,
            top_bottom_footprint_ratio=1.0,
            recurrence_count=1,
            composite_confidence=event.confidence,
        )
        risk_result = score_event(risk_input)

        # Confidence gate
        gate_input = ConfidenceInput(
            detector_confidence=event.confidence,
            track_stability=min(event.confidence * 1.2, 1.0),
            vlm_agreement=-1,
        )
        gate_result = apply_confidence_gate(gate_input)

        scored.append({
            "type": event.event_type.value,
            "risk_score": round(risk_result.score, 1),
            "tier": risk_result.tier,
            "status": gate_result.status,
            "should_alert": gate_result.should_alert,
            "composite_confidence": round(gate_result.composite_confidence, 3),
        })

        print(f"\n  {event.event_type.value}:")
        print(f"    Risk Score: {risk_result.score:.1f}/100 ({risk_result.tier})")
        print(f"    Confidence Gate: {gate_result.status} "
              f"(composite: {gate_result.composite_confidence:.3f})")
        print(f"    Alert: {'🔔 YES' if gate_result.should_alert else 'No'}")
        print(f"    Breakdown: {json.dumps({k: round(v,1) for k,v in risk_result.breakdown.items()})}")

    return scored


def test_feedback_loop(scored_events):
    """Test the human feedback + recalibration loop."""
    print("\n" + "=" * 60)
    print("Feedback Loop")
    print("=" * 60)

    for event in scored_events[:2]:
        old_status = event["status"]
        new_status = update_status_after_feedback(old_status, "confirm", event["composite_confidence"])
        print(f"  Confirm: {old_status} → {new_status} ({event['type']})")

    for event in scored_events[:1]:
        old_status = event["status"]
        new_status = update_status_after_feedback(old_status, "dismiss", event["composite_confidence"])
        print(f"  Dismiss: {old_status} → {new_status} ({event['type']})")


def main():
    print("ARGUS Full Pipeline Test — Realistic Simulated Detections")
    print("=" * 60)

    # Run all scenarios
    events1 = simulate_carry_and_drop()
    events2 = simulate_dragging()
    events3 = simulate_throwing()

    all_events = events1 + events2 + events3
    print(f"\n{'='*60}")
    print(f"TOTAL EVENTS ACROSS ALL SCENARIOS: {len(all_events)}")
    print(f"{'='*60}")

    # Feature extraction
    test_feature_extraction()

    # Risk scoring
    scored = test_risk_scoring_pipeline(all_events)

    # Feedback loop
    test_feedback_loop(scored)

    # Summary
    print("\n" + "=" * 60)
    print("FULL PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Scenarios tested: 3 (carry+drop, drag, throw)")
    print(f"  Total events: {len(all_events)}")
    print(f"  Events by type:")
    type_counts = {}
    for e in all_events:
        type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1
    for t, c in type_counts.items():
        print(f"    {t}: {c}")

    tier_counts = {}
    for s in scored:
        tier_counts[s["tier"]] = tier_counts.get(s["tier"], 0) + 1
    print(f"  Tier distribution: {tier_counts}")

    alerts = sum(1 for s in scored if s["should_alert"])
    print(f"  Alerts triggered: {alerts}/{len(scored)}")
    print("=" * 60)
    print("✅ Full pipeline test complete!")


if __name__ == "__main__":
    main()
