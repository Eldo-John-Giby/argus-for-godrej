"""Alert API routes — multilingual voice alert delivery for High/Critical events.

The frontend polls /api/events/ every 5 s and detects new High/Critical rows
itself; these endpoints supply the text, translation and audio, plus an SSE
stream for backends that want to push instead.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alerts import (
    SUPPORTED_LANGUAGES,
    compose_alert_text,
    parse_lang,
    render_sse,
    synthesize_speech,
    translate_text,
)
from app.database import get_db
from app.models import Camera, Event, Track

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


async def _load_event(db: AsyncSession, event_id: int) -> Optional[tuple[Event, Optional[str], Optional[str]]]:
    """Fetch the event plus its bay_id and camera name (for the alert line)."""
    result = await db.execute(
        select(Event, Camera.bay_id, Camera.name)
        .join(Track, Event.track_id == Track.id)
        .join(Camera, Track.camera_id == Camera.id)
        .where(Event.id == event_id)
    )
    return result.first()


@router.get("/languages")
async def list_languages():
    """Languages the alert can be delivered in (drives the UI selector)."""
    return [{"code": code, "name": name} for code, name in sorted(SUPPORTED_LANGUAGES.items())]


@router.get("/text")
async def alert_text(
    event_id: int = Query(...),
    lang: str = Query("en", description="Language code, e.g. hi, ta, hi-IN."),
    db: AsyncSession = Depends(get_db),
):
    """Composed (and optionally translated) alert text for one event."""
    row = await _load_event(db, event_id)
    if not row:
        return {"error": "event not found", "text": None}

    event, bay_id, camera_name = row
    base = compose_alert_text(
        behaviour_type=event.behaviour_type,
        tier=event.tier,
        risk_score=event.risk_score,
        bay_id=bay_id,
        camera_name=camera_name,
        timestamp=event.ts,
    )
    lang_code = parse_lang(lang)
    translated = await translate_text(base, lang_code)

    return {
        "event_id": event_id,
        "tier": event.tier,
        "language": translated["language"],
        "translated": translated["translated"],
        "text": translated["text"],
        "base_text": base,
    }


@router.get("/speak")
async def speak(
    event_id: int = Query(...),
    lang: str = Query("en", description="Language code, e.g. hi, ta, hi-IN."),
    db: AsyncSession = Depends(get_db),
):
    """Alert text as audio (gTTS). Returns 204 when TTS is unavailable —
    the frontend then speaks the /text response via browser SpeechSynthesis."""
    row = await _load_event(db, event_id)
    if not row:
        return Response(status_code=404)

    event, bay_id, camera_name = row
    base = compose_alert_text(
        behaviour_type=event.behaviour_type,
        tier=event.tier,
        risk_score=event.risk_score,
        bay_id=bay_id,
        camera_name=camera_name,
        timestamp=event.ts,
    )
    lang_code = parse_lang(lang)
    translated = await translate_text(base, lang_code)

    audio, content_type = await synthesize_speech(translated["text"], translated["language"])
    if not audio:
        return Response(status_code=204)

    return Response(content=audio, media_type=content_type or "audio/mpeg")


@router.get("/stream")
async def stream_alerts(
    lang: str = Query("en"),
    min_tier: str = Query("High", pattern="^(High|Critical|Medium|Low)$"),
    poll_seconds: float = Query(5, gt=0, le=60),
    db: AsyncSession = Depends(get_db),
):
    """SSE stream of new High/Critical events.

    Optional alternative to frontend polling — kept for completeness since
    the Live Feed already polls; the frontend currently detects new events
    from its own polling loop.
    """
    lang_code = parse_lang(lang)
    tier_rank = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    min_rank = tier_rank[min_tier]
    last_seen_id = 0

    async def event_stream():
        nonlocal last_seen_id
        while True:
            now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            result = await db.execute(
                select(Event)
                .where(Event.id > last_seen_id, Event.ts >= now - dt.timedelta(minutes=10))
                .order_by(Event.id)
            )
            for event in result.scalars().all():
                last_seen_id = max(last_seen_id, event.id)
                if tier_rank.get(event.tier, 0) < min_rank:
                    continue
                row = await _load_event(db, event.id)
                bay_id, camera_name = (row[1], row[2]) if row else (None, None)
                base = compose_alert_text(
                    behaviour_type=event.behaviour_type,
                    tier=event.tier,
                    risk_score=event.risk_score,
                    bay_id=bay_id,
                    camera_name=camera_name,
                    timestamp=event.ts,
                )
                translated = await translate_text(base, lang_code)
                yield render_sse({
                    "event_id": event.id,
                    "tier": event.tier,
                    "language": translated["language"],
                    "translated": translated["translated"],
                    "text": translated["text"],
                    "base_text": base,
                })
            await asyncio.sleep(poll_seconds)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
