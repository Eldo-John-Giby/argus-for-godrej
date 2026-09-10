"""Report API routes — automatic shift reports for the Prevention & Learning story."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.reports import build_shift_report

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/shift")
async def shift_report(
    hours: Optional[float] = Query(
        None, gt=0, description="Shift length in hours, counted back from `end_time` (default 8)."
    ),
    start_time: Optional[str] = Query(None, description="ISO timestamp; overrides `hours` when given."),
    end_time: Optional[str] = Query(None, description="ISO timestamp; defaults to now."),
    bay_id: Optional[str] = Query(None, description="Restrict the report to one bay."),
    format: str = Query("json", pattern="^(json|text)$"),
    db: AsyncSession = Depends(get_db),
):
    """Generate an automatic incident/shift report from stored events."""
    end = dt.datetime.fromisoformat(end_time) if end_time else dt.datetime.now(dt.timezone.utc)
    if start_time:
        start = dt.datetime.fromisoformat(start_time)
    else:
        start = end - dt.timedelta(hours=hours if hours is not None else 8)
    # SQLite stores naive datetimes; a tz-aware `end` with a naive `start`
    # would raise TypeError on comparison, so normalise to naive UTC.
    if end.tzinfo is not None:
        end = end.replace(tzinfo=None)
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)

    report = await build_shift_report(db, start_time=start, end_time=end, bay_id=bay_id)

    if format == "text":
        return PlainTextResponse(report["report_text"], media_type="text/plain")
    return report
