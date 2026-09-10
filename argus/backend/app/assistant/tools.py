"""Tool functions for the Supervisor AI Assistant.

These are the ONLY functions the assistant can call — it is explicitly
restricted to these DB-grounded tools and cannot generate information
outside of tool results.

Required tools per the brief's own example questions:
1. get_events(filters) → "Show me all high-risk handling events from today's unloading"
2. get_top_behaviours(period) → "What were the three most common risky behaviours?"
3. get_bay_summary(bay_id, shift) → "Which loading bay had the highest number of risky events?"
4. explain_event(event_id) → "Why was this event classified as high risk?"
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera, Event, Track, Feedback


async def get_events_tool(
    db: AsyncSession,
    behaviour_type: Optional[str] = None,
    bay_id: Optional[str] = None,
    status: Optional[str] = None,
    tier: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Query events from the database with optional filters.

    Use this to answer questions like:
    - "Show me all high-risk handling events from today's unloading"
    - "What events occurred in bay A this morning?"

    Returns JSON string of matching events.
    """
    query = select(Event).join(Track).where(True)

    if behaviour_type:
        query = query.where(Event.behaviour_type == behaviour_type)
    if status:
        query = query.where(Event.status == status)
    if tier:
        query = query.where(Event.tier == tier)
    if bay_id:
        query = query.join(Camera, Track.camera_id == Camera.id).where(Camera.bay_id == bay_id)
    if start_time:
        query = query.where(Event.ts >= dt.datetime.fromisoformat(start_time))
    if end_time:
        query = query.where(Event.ts <= dt.datetime.fromisoformat(end_time))

    query = query.order_by(Event.ts.desc()).limit(limit)
    result = await db.execute(query)
    events = result.scalars().all()

    return json.dumps([
        {
            "id": e.id,
            "behaviour_type": e.behaviour_type,
            "tier": e.tier,
            "risk_score": e.risk_score,
            "status": e.status,
            "confidence": e.confidence,
            "timestamp": e.ts.isoformat() if e.ts else None,
            "vlm_explanation": e.vlm_explanation,
        }
        for e in events
    ], indent=2)


async def get_top_behaviours_tool(
    db: AsyncSession,
    period: str = "today",
    limit: int = 5,
) -> str:
    """Get the most common risky behaviours in a time period.

    Use this to answer:
    - "What were the three most common risky behaviours during the morning shift?"
    - "Which behaviours are most frequent this week?"

    Returns JSON string of behaviour counts and average risk scores.
    """
    now = dt.datetime.now(dt.timezone.utc)

    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "morning_shift":
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    elif period == "evening_shift":
        start = now.replace(hour=14, minute=0, second=0, microsecond=0)
    else:
        start = now - dt.timedelta(days=7)

    query = (
        select(
            Event.behaviour_type,
            func.count(Event.id).label("count"),
            func.avg(Event.risk_score).label("avg_risk"),
        )
        .where(Event.ts >= start)
        .group_by(Event.behaviour_type)
        .order_by(func.count(Event.id).desc())
        .limit(limit)
    )
    result = await db.execute(query)
    rows = result.all()

    return json.dumps([
        {
            "behaviour_type": row[0],
            "count": row[1],
            "avg_risk_score": round(float(row[2] or 0), 1),
        }
        for row in rows
    ], indent=2)


async def get_bay_summary_tool(
    db: AsyncSession,
    bay_id: Optional[str] = None,
    shift: str = "today",
) -> str:
    """Get event summary for a loading bay.

    Use this to answer:
    - "Which loading bay had the highest number of risky events?"
    - "Show me bay 3's safety summary."

    Returns JSON string of bay event summary.
    """
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # If no specific bay, get all bays
    if bay_id:
        # Join through Track -> Camera to filter by bay_id
        query = (
            select(
                Event.tier,
                func.count(Event.id).label("count"),
                func.avg(Event.risk_score).label("avg_risk"),
            )
            .join(Track, Event.track_id == Track.id)
            .join(Camera, Track.camera_id == Camera.id)
            .where(Event.ts >= start, Camera.bay_id == bay_id)
            .group_by(Event.tier)
        )
        result = await db.execute(query)
        tiers = {row[0]: row[1] for row in result.all()}
        total = sum(tiers.values())
        avg_risk_result = await db.execute(
            select(func.avg(Event.risk_score))
            .join(Track, Event.track_id == Track.id)
            .join(Camera, Track.camera_id == Camera.id)
            .where(Event.ts >= start, Camera.bay_id == bay_id)
        )
        avg_risk = avg_risk_result.scalar() or 0

        return json.dumps({
            "bay_id": bay_id,
            "total_events": total,
            "tier_breakdown": tiers,
            "average_risk_score": round(float(avg_risk), 1),
        }, indent=2)
    else:
        # Get summary for all bays
        query = (
            select(
                Track.camera_id,
                func.count(Event.id).label("count"),
                func.avg(Event.risk_score).label("avg_risk"),
            )
            .join(Event, Track.id == Event.track_id)
            .where(Event.ts >= start)
            .group_by(Track.camera_id)
            .order_by(func.count(Event.id).desc())
        )
        result = await db.execute(query)
        bays = [
            {
                "bay_id": str(row[0]),
                "event_count": row[1],
                "avg_risk": round(float(row[2] or 0), 1),
            }
            for row in result.all()
        ]

        return json.dumps({
            "period": shift,
            "bays": bays,
            "highest_risk_bay": bays[0] if bays else None,
        }, indent=2)


async def explain_event_tool(
    db: AsyncSession,
    event_id: int,
) -> str:
    """Explain why an event was classified with its risk level.

    Use this to answer:
    - "Why was this event classified as high risk?"
    - "Walk me through the risk breakdown for event 42."

    Returns JSON string with full risk breakdown and VLM explanation.
    """
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()

    if not event:
        return json.dumps({"error": f"Event {event_id} not found"})

    # Get feedback count for this event
    feedback_result = await db.execute(
        select(func.count(Feedback.id)).where(Feedback.event_id == event_id)
    )
    feedback_count = feedback_result.scalar() or 0

    return json.dumps({
        "event_id": event.id,
        "behaviour_type": event.behaviour_type,
        "risk_score": event.risk_score,
        "tier": event.tier,
        "confidence": event.confidence,
        "status": event.status,
        "features": event.features_json or {},
        "vlm_explanation": event.vlm_explanation or "No VLM explanation available",
        "feedback_count": feedback_count,
        "risk_breakdown": (event.features_json or {}).get("risk_breakdown", {}),
        "explanation": (
            f"This event was classified as {event.tier} risk "
            f"(score: {event.risk_score}/100) because the system detected "
            f"behaviour '{event.behaviour_type}' with {event.confidence:.0%} confidence. "
            f"{event.vlm_explanation or ''}"
        ),
    }, indent=2)
