"""Events CRUD and evidence storage."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Camera, Event, Track


async def create_event(
    db: AsyncSession,
    track_id: int,
    behaviour_type: str,
    timestamp: dt.datetime,
    features: dict,
    risk_score: float,
    tier: str,
    confidence: float,
    status: str,
    clip_path: Optional[str] = None,
    keyframe_path: Optional[str] = None,
    vlm_explanation: Optional[str] = None,
    risk_breakdown: Optional[dict] = None,
) -> Event:
    """Create a new event record."""
    # Persist risk_breakdown inside features_json so it's available on read
    merged_features = dict(features) if features else {}
    if risk_breakdown:
        merged_features["risk_breakdown"] = risk_breakdown

    event = Event(
        track_id=track_id,
        behaviour_type=behaviour_type,
        ts=timestamp,
        features_json=merged_features,
        risk_score=risk_score,
        tier=tier,
        confidence=confidence,
        status=status,
        clip_path=clip_path,
        keyframe_path=keyframe_path,
        vlm_explanation=vlm_explanation,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return event


async def get_event(db: AsyncSession, event_id: int) -> Optional[Event]:
    """Get a single event by ID."""
    result = await db.execute(select(Event).where(Event.id == event_id))
    return result.scalar_one_or_none()


async def get_events(
    db: AsyncSession,
    behaviour_type: Optional[str] = None,
    bay_id: Optional[str] = None,
    status: Optional[str] = None,
    tier: Optional[str] = None,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Event]:
    """Query events with optional filters."""
    query = select(Event).join(Track).where(True)

    if behaviour_type:
        query = query.where(Event.behaviour_type == behaviour_type)
    if status:
        query = query.where(Event.status == status)
    if tier:
        query = query.where(Event.tier == tier)
    if start_time:
        query = query.where(Event.ts >= start_time)
    if end_time:
        query = query.where(Event.ts <= end_time)
    if bay_id:
        query = query.join(Camera, Track.camera_id == Camera.id).where(Camera.bay_id == bay_id)

    query = query.order_by(Event.ts.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_top_behaviours(
    db: AsyncSession,
    start_time: dt.datetime,
    end_time: dt.datetime,
    limit: int = 10,
) -> list[dict]:
    """Get most common behaviours in a time window."""
    query = (
        select(
            Event.behaviour_type,
            func.count(Event.id).label("count"),
            func.avg(Event.risk_score).label("avg_risk"),
        )
        .where(Event.ts >= start_time, Event.ts <= end_time)
        .group_by(Event.behaviour_type)
        .order_by(func.count(Event.id).desc())
        .limit(limit)
    )
    result = await db.execute(query)
    return [
        {"behaviour_type": row[0], "count": row[1], "avg_risk": float(row[2] or 0)}
        for row in result.all()
    ]


async def get_bay_summary(
    db: AsyncSession,
    bay_id: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
) -> dict:
    """Get event summary for a specific bay."""
    events = await get_events(db, bay_id=bay_id, start_time=start_time, end_time=end_time)
    tier_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for e in events:
        tier_counts[e.tier] = tier_counts.get(e.tier, 0) + 1

    return {
        "bay_id": bay_id,
        "total_events": len(events),
        "tier_counts": tier_counts,
        "avg_risk_score": sum(e.risk_score for e in events) / len(events) if events else 0,
    }


async def explain_event(db: AsyncSession, event_id: int) -> Optional[dict]:
    """Get detailed explanation of why an event was flagged."""
    event = await get_event(db, event_id)
    if not event:
        return None

    features = event.features_json or {}
    risk_breakdown = features.get("risk_breakdown", {})

    # Generate human-readable explanation
    behaviour_label = event.behaviour_type.replace("_", " ")
    explanation = (
        f"This event was classified as {event.tier} risk "
        f"(score: {event.risk_score}/100) because the system detected "
        f"behaviour '{behaviour_label}' with {event.confidence:.0%} confidence."
    )
    if event.vlm_explanation:
        explanation += f" {event.vlm_explanation}"

    return {
        "event_id": event.id,
        "behaviour_type": event.behaviour_type,
        "risk_score": event.risk_score,
        "tier": event.tier,
        "confidence": event.confidence,
        "features": features,
        "vlm_explanation": event.vlm_explanation,
        "risk_breakdown": risk_breakdown,
        "explanation": explanation,
        "status": event.status,
        "timestamp": event.ts.isoformat() if event.ts else None,
        "clip_path": event.clip_path,
        "clip_url": _clip_url(event.clip_path),
    }


def _clip_url(clip_path: Optional[str]) -> Optional[str]:
    """Public URL for an event clip, served by the backend's /clips mount.

    Seeded rows store '<video_slug>/<clip>.mp4' relative to the clips root.
    Legacy placeholder paths ('clips/<behaviour>.mp4' from older seeds) map
    to no real file and return None so the frontend shows its fallback.
    """
    if not clip_path or not clip_path.lower().endswith(".mp4"):
        return None
    if clip_path.startswith("clips/"):
        return None
    return f"/clips/{clip_path}"


async def update_event_status(
    db: AsyncSession,
    event_id: int,
    status: str,
) -> Optional[Event]:
    """Update event status (observed/potential/confirmed/dismissed)."""
    event = await get_event(db, event_id)
    if not event:
        return None
    event.status = status
    await db.commit()
    await db.refresh(event)
    return event
