"""Bay aggregation API routes — heatmap data and bay summaries."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Camera, Event, Track

router = APIRouter(prefix="/api/bays", tags=["bays"])


@router.get("/")
async def list_bays(db: AsyncSession = Depends(get_db)):
    """Get all bays with their event counts and risk aggregation.

    Returns list of bay summaries for the heatmap page.
    """
    now = dt.datetime.now(dt.timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get all cameras grouped by bay_id
    cameras_result = await db.execute(
        select(Camera.bay_id, Camera.id).distinct()
    )
    camera_map = {}  # bay_id -> [camera_ids]
    for bay_id, cam_id in cameras_result.all():
        if bay_id not in camera_map:
            camera_map[bay_id] = []
        camera_map[bay_id].append(cam_id)

    bays = []
    for bay_id, camera_ids in camera_map.items():
        # Count events per tier for this bay
        events_query = (
            select(
                Event.tier,
                func.count(Event.id).label("count"),
                func.avg(Event.risk_score).label("avg_risk"),
            )
            .join(Track, Event.track_id == Track.id)
            .where(
                Track.camera_id.in_(camera_ids),
                Event.ts >= start_of_day,
            )
            .group_by(Event.tier)
        )
        result = await db.execute(events_query)
        tier_data = {row[0]: {"count": row[1], "avg_risk": float(row[2] or 0)} for row in result.all()}

        total_events = sum(d["count"] for d in tier_data.values())
        avg_risk = (
            sum(d["avg_risk"] * d["count"] for d in tier_data.values()) / total_events
            if total_events > 0
            else 0
        )

        bays.append({
            "bay_id": bay_id,
            "event_count": total_events,
            "avg_risk": round(avg_risk, 1),
            "critical": tier_data.get("Critical", {}).get("count", 0),
            "high": tier_data.get("High", {}).get("count", 0),
            "medium": tier_data.get("Medium", {}).get("count", 0),
            "low": tier_data.get("Low", {}).get("count", 0),
        })

    # Sort by event count descending
    bays.sort(key=lambda b: b["event_count"], reverse=True)
    return bays


@router.get("/operators")
async def list_operators(db: AsyncSession = Depends(get_db)):
    """Get per-operator aggregate stats from event data.

    Groups by track (person) and computes event counts, risk scores.
    Since we don't have an operator model, uses track_id as proxy.
    """
    now = dt.datetime.now(dt.timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Aggregate events per track (proxy for operator)
    query = (
        select(
            Track.id.label("track_id"),
            Track.object_class,
            func.count(Event.id).label("events_total"),
            func.avg(Event.risk_score).label("avg_risk"),
        )
        .join(Event, Track.id == Event.track_id)
        .where(Event.ts >= start_of_day, Track.object_class == "person")
        .group_by(Track.id, Track.object_class)
        .order_by(func.count(Event.id).desc())
        .limit(20)
    )
    result = await db.execute(query)

    operators = []
    for i, row in enumerate(result.all()):
        track_id, obj_class, events_total, avg_risk = row
        operators.append({
            "operator_id": f"OP-{track_id:03d}",
            "name": f"Operator #{track_id}",
            "role": "Operator",
            "events_total": events_total,
            "confirms": 0,  # Would need feedback join
            "dismissals": 0,
            "false_positive_rate": 0.0,
            "avg_risk": round(float(avg_risk or 0), 1),
            "shift": "Morning" if now.hour < 14 else "Evening",
        })

    return operators
