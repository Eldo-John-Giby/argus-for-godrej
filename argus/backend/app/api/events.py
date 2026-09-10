"""Events API routes — CRUD, querying, evidence storage.

Route ordering matters: specific routes (top-behaviours, bay-summary, explain)
must be defined BEFORE the /{event_id} catch-all, otherwise FastAPI matches
them as event_id="top-behaviours" and returns 422.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.events import get_event, get_events, get_top_behaviours, get_bay_summary, explain_event

router = APIRouter(prefix="/api/events", tags=["events"])


# --- Specific routes FIRST (before /{event_id} catch-all) ---


@router.get("/top-behaviours")
async def top_behaviours(
    start_time: str = Query(...),
    end_time: str = Query(...),
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Get most common behaviours in a time window."""
    return await get_top_behaviours(
        db,
        start_time=dt.datetime.fromisoformat(start_time),
        end_time=dt.datetime.fromisoformat(end_time),
        limit=limit,
    )


@router.get("/bay-summary/{bay_id}")
async def bay_summary(
    bay_id: str,
    start_time: str = Query(...),
    end_time: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get event summary for a specific bay."""
    return await get_bay_summary(
        db, bay_id,
        start_time=dt.datetime.fromisoformat(start_time),
        end_time=dt.datetime.fromisoformat(end_time),
    )


@router.get("/explain/{event_id}")
async def explain(event_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed explanation of why an event was flagged."""
    explanation = await explain_event(db, event_id)
    if not explanation:
        raise HTTPException(status_code=404, detail="Event not found")
    return explanation


# --- Generic routes ---


@router.get("/")
async def list_events(
    behaviour_type: Optional[str] = Query(None),
    bay_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List events with optional filters."""
    start = dt.datetime.fromisoformat(start_time) if start_time else None
    end = dt.datetime.fromisoformat(end_time) if end_time else None

    events = await get_events(
        db, behaviour_type=behaviour_type, bay_id=bay_id,
        status=status, tier=tier, start_time=start, end_time=end,
        limit=limit, offset=offset,
    )
    return [
        {
            "id": e.id,
            "behaviour_type": e.behaviour_type,
            "tier": e.tier,
            "risk_score": e.risk_score,
            "status": e.status,
            "confidence": e.confidence,
            "timestamp": e.ts.isoformat() if e.ts else None,
            "clip_path": e.clip_path,
            "keyframe_path": e.keyframe_path,
            "vlm_explanation": e.vlm_explanation,
        }
        for e in events
    ]


@router.get("/{event_id}")
async def get_event_detail(event_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed event information."""
    event = await get_event(db, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    explanation = await explain_event(db, event_id)
    return explanation
