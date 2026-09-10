"""Automatic incident/shift reports.

Pulls a shift's events from the DB, groups them by behaviour and bay, and
renders a formatted summary with a recommended action per behaviour.
Zero new infrastructure — templating over data that already exists.

Design rule: every number in the report is derived from DB rows. No
fabricated "insight" text — recommendations come from a fixed lookup keyed
by behaviour type so the report stays defensible.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera, Event, Feedback, Track

# Fixed recommendation per behaviour — text a supervisor can act on, not prose.
RECOMMENDED_ACTIONS = {
    "1_product_dropped": "Re-brief correct lifting technique; check grip and load weight limits",
    "2_product_dragged": "Re-brief handling procedure; provide trolleys at point of use",
    "3_product_thrown": "Immediate corrective briefing; escalating discipline for repeat offences",
    "4_improper_stacking": "Review stacking procedure with team; mark max stack height on floor",
    "5_unstable_stacking": "Review stacking procedure with team; check pallet condition",
    "6_stepping_on_packages": "Provide step platforms; re-brief no-step-on-carton rule",
    "7_wrong_zone": "Re-mark floor zoning; re-brief drop-off locations",
    "8_dragging_no_trolley": "Audit trolley availability and placement in this bay",
    "11_wrong_orientation": "Check orientation labels visibility and re-brief handling marks",
}

DEFAULT_ACTION = "Review bay CCTV for the shift and re-brief the responsible team"

TIER_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _label(behaviour_type: str) -> str:
    """'3_product_thrown' -> 'product thrown'."""
    return behaviour_type.replace("_", " ").strip()


def _recommended_action(behaviour_type: str) -> str:
    base = behaviour_type.lower()
    for key, action in RECOMMENDED_ACTIONS.items():
        if key in base or base.endswith(key.split("_", 1)[-1]):
            return action
    return DEFAULT_ACTION


async def build_shift_report(
    db: AsyncSession,
    start_time: dt.datetime,
    end_time: Optional[dt.datetime] = None,
    bay_id: Optional[str] = None,
    title: str = "Shift report",
) -> dict:
    """Aggregate a shift's events into a formatted summary.

    Returns a dict with structured counts plus a ready-to-display text body
    and a suggested filename for download.
    """
    end_time = end_time or dt.datetime.now(dt.timezone.utc)

    # --- events in the window (join Track -> Camera for bay + camera name) ---
    query = (
        select(
            Event,
            Track.camera_id,
            Camera.bay_id,
            Camera.name,
        )
        .join(Track, Event.track_id == Track.id)
        .join(Camera, Track.camera_id == Camera.id)
        .where(Event.ts >= start_time, Event.ts <= end_time)
    )
    if bay_id:
        query = query.where(Camera.bay_id == bay_id)

    result = await db.execute(query)
    rows = result.all()

    events = rows
    total = len(events)

    # --- by tier ---
    tier_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for ev, *_ in events:
        tier_counts[ev.tier] = tier_counts.get(ev.tier, 0) + 1

    # --- by bay (uses camera names as labels when present) ---
    by_bay: dict[str, dict] = {}
    for ev, _cam_id, b_id, cam_name in events:
        entry = by_bay.setdefault(
            b_id, {"bay_id": b_id, "label": cam_name or f"Bay {b_id}", "total": 0, "high_or_worse": 0}
        )
        entry["total"] += 1
        if ev.tier in ("Critical", "High"):
            entry["high_or_worse"] += 1
    bays = sorted(by_bay.values(), key=lambda b: (-b["high_or_worse"], -b["total"]))

    # --- by behaviour ---
    by_behaviour: dict[str, dict] = {}
    for ev, *_ in events:
        entry = by_behaviour.setdefault(
            ev.behaviour_type,
            {"behaviour_type": ev.behaviour_type, "label": _label(ev.behaviour_type), "count": 0, "avg_risk": 0.0},
        )
        entry["count"] += 1
        entry["avg_risk"] += ev.risk_score
    behaviours = sorted(by_behaviour.values(), key=lambda b: (-b["count"], -b["avg_risk"]))
    for b in behaviours:
        b["avg_risk"] = round(b["avg_risk"] / b["count"], 1)

    # --- feedback / false-positive rate in the same window ---
    fb_query = (
        select(Feedback.action, func.count(Feedback.id))
        .join(Event, Feedback.event_id == Event.id)
        .where(Feedback.ts >= start_time, Feedback.ts <= end_time)
        .group_by(Feedback.action)
    )
    if bay_id:
        fb_query = fb_query.join(Track, Event.track_id == Track.id).join(
            Camera, Track.camera_id == Camera.id
        ).where(Camera.bay_id == bay_id)
    fb_result = await db.execute(fb_query)
    fb_counts = {"confirm": 0, "dismiss": 0}
    for action, count in fb_result.all():
        fb_counts[action] = fb_counts.get(action, 0) + 1
    fb_total = fb_counts["confirm"] + fb_counts["dismiss"]
    false_positive_rate = (fb_counts["dismiss"] / fb_total) if fb_total else None

    top_behaviour = behaviours[0]["label"] if behaviours else None

    report = {
        "title": title,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "bay_id": bay_id,
        "total_events": total,
        "tier_counts": tier_counts,
        "bays": bays,
        "behaviours": behaviours,
        "top_behaviour": top_behaviour,
        "feedback": {
            "confirms": fb_counts["confirm"],
            "dismissals": fb_counts["dismiss"],
            "total": fb_total,
            "false_positive_rate": false_positive_rate,
        },
    }

    report["report_text"] = _render_text(report)
    report["filename"] = _filename(report)
    return report


def _render_text(r: dict) -> str:
    """Plain-text body — displayed in the dashboard card and used for TTS."""
    lines: list[str] = []
    scope = f"Bay {r['bay_id']}" if r["bay_id"] else "All bays"
    when = _fmt_day(r["start_time"])
    lines.append(f"{r['title']} — {scope}, {when}")

    tiers = r["tier_counts"]
    lines.append(f"{r['total_events']} events: {tiers['Critical']} critical, "
                 f"{tiers['High']} high, {tiers['Medium']} medium, {tiers['Low']} low.")

    for b in r["bays"][:3]:
        lines.append(f"{b['label']}: {b['total']} events, {b['high_or_worse']} high or critical.")

    if r["behaviours"]:
        top = r["behaviours"][0]
        others = [b for b in r["behaviours"][1:4]]
        parts = ", ".join(f"{b['label']} ({b['count']}x)" for b in others)
        line = f"Top behaviour: {top['label']} ({top['count']}x)"
        if parts:
            line += f"; also {parts}"
        line += "."
        lines.append(line)

    # One recommended action for the top behaviour, one for the worst bay.
    if r["behaviours"]:
        lines.append(f"Recommended action: {_recommended_action(r['behaviours'][0]['behaviour_type'])}.")
    if r["bays"] and r["bays"][0]["high_or_worse"] > 0:
        lines.append(f"Priority bay: {r['bays'][0]['label']} "
                     f"({r['bays'][0]['high_or_worse']} high/critical events).")

    fb = r["feedback"]
    if fb["total"] > 0 and fb["false_positive_rate"] is not None:
        lines.append(f"Supervisor review: {fb['total']} events reviewed, "
                     f"false-positive rate {fb['false_positive_rate'] * 100:.0f}%.")

    return "\n".join(lines)


def _fmt_day(iso_ts: str) -> str:
    try:
        d = dt.datetime.fromisoformat(iso_ts)
        return d.strftime("%b %-d")  # e.g. "Sep 8"
    except (ValueError, TypeError):
        return iso_ts[:10]


def _filename(r: dict) -> str:
    scope = r["bay_id"] or "all-bays"
    day = _fmt_day(r["start_time"]).replace(" ", "-").lower()
    return f"argus-shift-report-{scope}-{day}.txt"
