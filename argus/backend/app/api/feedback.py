"""Feedback API routes — human confirm/dismiss + recalibration."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.feedback import submit_feedback, get_feedback_stats, recalibrate_thresholds

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    event_id: int
    reviewer: str
    action: str  # "confirm" | "dismiss"


@router.post("/")
async def create_feedback(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    """Submit human feedback on an event (confirm or dismiss)."""
    if req.action not in ("confirm", "dismiss"):
        raise HTTPException(status_code=400, detail="Action must be 'confirm' or 'dismiss'")

    result = await submit_feedback(db, req.event_id, req.reviewer, req.action)
    if not result:
        raise HTTPException(status_code=404, detail="Event not found")
    return result


@router.get("/stats")
async def feedback_stats(
    behaviour_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get feedback statistics for recalibration analysis."""
    start = dt.datetime.fromisoformat(start_time) if start_time else None
    end = dt.datetime.fromisoformat(end_time) if end_time else None
    return await get_feedback_stats(db, behaviour_type=behaviour_type, start_time=start, end_time=end)


@router.post("/recalibrate")
async def trigger_recalibration(db: AsyncSession = Depends(get_db)):
    """Trigger threshold recalibration based on accumulated feedback.

    For demo: click a button to show before/after threshold numbers.
    """
    return await recalibrate_thresholds(db)
