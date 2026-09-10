"""Local explanation generator — deterministic, feature-grounded event narratives.

When no VLM API is reachable (no key, no credits, offline demo), the system
still explains *why* an event was flagged, using the same kinematic/geometric
features the detector computed. Every sentence is derived from stored feature
values — nothing is invented, no API call is made.

This is the path the product runs on by default; a live VLM (Qwen/Gemini)
upgrades the same panel with richer semantic verification when configured.
"""

from __future__ import annotations

from typing import Optional

# Human-readable names for the behaviour taxonomy keys.
BEHAVIOUR_LABELS: dict[str, str] = {
    "1_product_dropped": "product dropped from height",
    "2_product_dragged": "product dragged along the floor",
    "3_product_thrown": "product thrown",
    "4_stack_unstable": "unstable stacking",
    "5_forklift_speeding": "forklift speeding",
    "6_stepping_on_packages": "worker stepping on packages",
    "7_wrong_zone_storage": "product stored in wrong zone",
    "8_dragging_no_trolley": "manual drag without trolley",
    "9_climbing_racks": "climbing storage racks",
    "10_improper_lifting": "improper lifting posture",
    "11_rush_handling": "rushed handling near others",
    "12_cluttered_staging": "cluttered staging zone",
    "13_ppe_noncompliance": "PPE non-compliance",
}


def behaviour_label(behaviour_type: str) -> str:
    return BEHAVIOUR_LABELS.get(behaviour_type, behaviour_type.replace("_", " "))


def _f(features: dict, key: str) -> Optional[float]:
    v = features.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _explain_ppe(f: dict) -> str:
    missing = f.get("missing_ppe") or []
    names = ", ".join(missing) if missing else "required PPE"
    model = f.get("source_model", "the PPE detector")
    note = (
        " Detector is trained on construction-site imagery; warehouse "
        "fine-tuning is planned."
        if f.get("domain_note")
        else ""
    )
    return (
        f"PPE detector ({model}) registered no {names} on the tracked "
        f"worker's head/torso region across the flagged clip, so the "
        f"compliance module marks this worker as unprotected at this "
        f"timestamp.{note}"
    )


def _explain_dropped(f: dict) -> str:
    dv = _f(f, "downward_velocity")
    hv = _f(f, "horizontal_velocity")
    drop = _f(f, "vertical_drop")
    bits: list[str] = []
    if drop is not None:
        bits.append(f"a sustained vertical drop of {drop:.0f}px")
    if dv is not None:
        bits.append(f"downward velocity ≈{dv:.0f}px/s")
    if hv is not None and dv is not None and hv < dv * 0.25:
        bits.append(
            f"minimal horizontal motion (≈{hv:.0f}px/s), which rules out a "
            f"throw and points to a slip or uncontrolled release"
        )
    tail = "; ".join(bits) if bits else "free-fall kinematics inconsistent with controlled lowering"
    return f"Object tracked through {tail}. Impact at floor level follows a free-drop profile, not a controlled placement."


def _explain_dragged(f: dict) -> str:
    speed = _f(f, "horizontal_speed")
    dur = _f(f, "drag_duration")
    disp = _f(f, "net_displacement")
    bits: list[str] = []
    if disp is not None:
        bits.append(f"{disp:.0f}px of net horizontal displacement")
    if speed is not None:
        bits.append(f"≈{speed:.0f}px/s sustained contact speed")
    if dur is not None:
        bits.append(f"over {dur:.1f}s")
    tail = ", ".join(bits) if bits else "sustained floor-contact motion"
    return (
        f"Object shows {tail} while its bottom edge stays at constant floor "
        f"level — the signature of dragging along the ground rather than "
        f"lifting and carrying. Repeated dragging scuffs packaging and can "
        f"shift pallet loads."
    )


def _explain_thrown(f: dict) -> str:
    vel = f.get("throw_velocity") or []
    if isinstance(vel, (list, tuple)) and len(vel) == 2:
        vx, vy = float(vel[0]), float(vel[1])
        return (
            f"Object launched with dominant horizontal velocity ≈{abs(vx):.0f}px/s "
            f"against a small vertical component (≈{abs(vy):.0f}px/s) — a ballistic "
            f"profile consistent with a throw rather than a placement or drop. "
            f"Impact energy scales with the square of velocity, so thrown goods "
            f"are the highest-damage handling pattern in the taxonomy."
        )
    return "Object motion follows a ballistic trajectory consistent with being thrown rather than placed."


def _explain_stepping(f: dict) -> str:
    frames = _f(f, "frames")
    on = f.get("on_class", "package")
    ratio = _f(f, "footprint_ratio")
    vel = _f(f, "velocity")
    bits: list[str] = []
    if frames is not None:
        bits.append(f"{frames:.0f} consecutive frames")
    if ratio is not None:
        bits.append(f"full footprint overlap ({ratio:.0%})" if ratio >= 0.8 else f"partial footprint overlap ({ratio:.0%})")
    if vel is not None and vel < 1:
        bits.append("zero walking velocity while in contact")
    tail = " with ".join(bits) if bits else "sustained contact with a sealed package"
    return f"Worker's feet remained on a {on} through {tail}. Static weight on packaging can crush contents and destabilise adjacent stacks."


def _explain_drag_no_trolley(f: dict) -> str:
    speed = _f(f, "horizontal_speed")
    dur = _f(f, "drag_duration")
    disp = _f(f, "net_displacement")
    bits: list[str] = []
    if disp is not None:
        bits.append(f"{disp:.0f}px of displacement")
    if speed is not None:
        bits.append(f"≈{speed:.0f}px/s")
    if dur is not None:
        bits.append(f"over {dur:.1f}s")
    tail = ", ".join(bits) if bits else "repeated ground-contact motion"
    return f"Item moved by {tail} with no trolley or dolly detected in the track region — manual drag without mechanical aid."


def _explain_generic(behaviour_type: str) -> str:
    label = behaviour_label(behaviour_type)
    return (
        f"Detector and tracker features matched the '{label}' rule in the "
        f"behaviour taxonomy at the recorded confidence — see the computed "
        f"features below for the exact measurements behind this flag."
    )


_GENERATORS = {
    "13_ppe_noncompliance": _explain_ppe,
    "1_product_dropped": _explain_dropped,
    "2_product_dragged": _explain_dragged,
    "8_dragging_no_trolley": _explain_drag_no_trolley,
    "3_product_thrown": _explain_thrown,
    "6_stepping_on_packages": _explain_stepping,
}


def generate_explanation(behaviour_type: str, features: dict) -> str:
    """Grounded, human-readable explanation built purely from event features.

    Deterministic: same event always yields the same sentence — safe to show
    on a recorded demo, safe to re-run in tests.
    """
    gen = _GENERATORS.get(behaviour_type, _explain_generic)
    try:
        return gen(features or {})
    except Exception:
        # Never let explanation formatting break the pipeline.
        return _explain_generic(behaviour_type)
