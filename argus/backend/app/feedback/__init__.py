"""Human feedback endpoint and recalibration job."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Event, Feedback
from app.risk.confidence_gate import update_status_after_feedback


async def submit_feedback(
    db: AsyncSession,
    event_id: int,
    reviewer: str,
    action: str,  # "confirm" | "dismiss"
) -> Optional[dict]:
    """Submit human feedback on an event and update its status.

    This is the feedback loop — the single highest-leverage differentiator.
    Every confirm/dismiss writes to the feedback table and can trigger
    threshold recalibration.
    """
    event_result = await db.execute(select(Event).where(Event.id == event_id))
    event = event_result.scalar_one_or_none()
    if not event:
        return None

    # Capture the old status BEFORE mutating
    old_status = event.status

    # Record the feedback
    feedback = Feedback(
        event_id=event_id,
        reviewer=reviewer,
        action=action,
        ts=dt.datetime.now(dt.timezone.utc),
    )
    db.add(feedback)

    # Update event status based on feedback
    new_status = update_status_after_feedback(
        current_status=event.status,
        action=action,
        composite_confidence=event.confidence,
    )
    event.status = new_status

    await db.commit()

    return {
        "event_id": event_id,
        "action": action,
        "old_status": old_status,
        "new_status": new_status,
        "reviewer": reviewer,
    }


async def get_feedback_stats(
    db: AsyncSession,
    behaviour_type: Optional[str] = None,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
) -> dict:
    """Get feedback statistics for recalibration analysis."""
    query = (
        select(
            Feedback.action,
            func.count(Feedback.id).label("count"),
        )
    )

    if behaviour_type or start_time or end_time:
        query = query.join(Event, Feedback.event_id == Event.id)
        if behaviour_type:
            query = query.where(Event.behaviour_type == behaviour_type)
        if start_time:
            query = query.where(Feedback.ts >= start_time)
        if end_time:
            query = query.where(Feedback.ts <= end_time)

    query = query.group_by(Feedback.action)
    result = await db.execute(query)

    stats = {"confirm": 0, "dismiss": 0}
    for row in result.all():
        stats[row[0]] = row[1]

    total = stats["confirm"] + stats["dismiss"]
    false_positive_rate = stats["dismiss"] / total if total > 0 else 0.0

    return {
        "confirms": stats["confirm"],
        "dismissals": stats["dismiss"],
        "total": total,
        "false_positive_rate": false_positive_rate,
    }


async def recalibrate_thresholds(
    db: AsyncSession,
) -> dict:
    """Recalibrate severity thresholds based on accumulated feedback.

    In production, this would run nightly. For demo, triggered on-demand.
    Analyzes feedback patterns and suggests threshold adjustments.

    Shows before/after numbers on screen — cheapest, highest-impact demo beat.
    """
    # Get overall feedback stats
    stats = await get_feedback_stats(db)

    # No feedback yet — nothing to recalibrate
    if stats["total"] == 0:
        return {
            "false_positive_rate": 0.0,
            "total_feedback": 0,
            "adjustment_direction": "no_data",
            "confidence_threshold_adjustment": 0.0,
            "per_behaviour": {},
            "message": "No feedback submitted yet. Confirm or dismiss events to enable recalibration.",
        }

    # Compute adjustment factor
    fpr = stats["false_positive_rate"]

    # If FPR is high (>0.3), reduce severity bases across the board
    # If FPR is low (<0.1), consider increasing sensitivity
    adjustment = {
        "false_positive_rate": fpr,
        "total_feedback": stats["total"],
        "adjustment_direction": "increase_sensitivity" if fpr < 0.1 else "decrease_sensitivity" if fpr > 0.3 else "stable",
        "confidence_threshold_adjustment": -0.05 if fpr > 0.3 else 0.05 if fpr < 0.1 else 0.0,
    }

    # Per-behaviour recalibration — single grouped query instead of N+1
    per_behaviour_query = (
        select(
            Event.behaviour_type,
            Feedback.action,
            func.count(Feedback.id).label("count"),
        )
        .join(Event, Feedback.event_id == Event.id)
        .group_by(Event.behaviour_type, Feedback.action)
    )
    result = await db.execute(per_behaviour_query)
    rows = result.all()

    # Aggregate into per-behaviour stats
    behaviour_stats: dict[str, dict[str, int]] = {}
    for behaviour_type, action, count in rows:
        if behaviour_type not in behaviour_stats:
            behaviour_stats[behaviour_type] = {"confirm": 0, "dismiss": 0}
        behaviour_stats[behaviour_type][action] = count

    per_behaviour = {}
    for b, b_data in behaviour_stats.items():
        b_total = b_data["confirm"] + b_data["dismiss"]
        if b_total >= 5:  # Only recalibrate with enough data
            b_fpr = b_data["dismiss"] / b_total
            per_behaviour[b] = {
                "fpr": b_fpr,
                "samples": b_total,
                "adjustment": "tighten" if b_fpr > 0.3 else "loosen" if b_fpr < 0.1 else "hold",
            }

    adjustment["per_behaviour"] = per_behaviour

    return adjustment
