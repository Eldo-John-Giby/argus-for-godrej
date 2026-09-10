"""Data-retention worker — enforces the RETENTION_DAYS policy.

The problem statement's responsible-AI section requires explicit footage
retention limits. Events older than RETENTION_DAYS (and their stored clip
files) are purged periodically by a background asyncio task started in
main.py's lifespan. RETENTION_DAYS=0 disables purging entirely (e.g. for a
demo dataset you want to keep).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from app.config import CLIPS_DIR, RETENTION_DAYS, RETENTION_SWEEP_HOURS

logger = logging.getLogger("argus.retention")


async def purge_expired(*, dry_run: bool = False) -> dict[str, Any]:
    """Delete events older than RETENTION_DAYS and unlink their clip files.

    Clip files are only removed when no remaining event references them
    (clips can be shared between events). Returns stats about what was —
    or with ``dry_run`` — would be removed.
    """
    if RETENTION_DAYS <= 0:
        return {"disabled": True, "events": 0, "clips": 0}

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=RETENTION_DAYS)

    from app.database import async_session
    from app.models import Event

    async with async_session() as session:
        rows = (await session.execute(
            select(Event.id, Event.clip_path).where(Event.ts < cutoff)
        )).all()
        if not rows:
            return {"disabled": False, "events": 0, "clips": 0, "cutoff": cutoff.isoformat()}

        # Reference count every clip so shared files survive one delete.
        clip_counts: dict[str, int] = {}
        all_clips = (await session.execute(
            select(Event.clip_path).where(Event.clip_path.isnot(None))
        )).all()
        for (cp,) in all_clips:
            if cp:
                clip_counts[cp] = clip_counts.get(cp, 0) + 1

        expired_ids = [rid for rid, _ in rows]
        if not dry_run:
            await session.execute(delete(Event).where(Event.id.in_(expired_ids)))
            await session.commit()

    removed_clips = 0
    if not dry_run:
        clips_root = CLIPS_DIR.resolve()
        for _, clip_path in rows:
            if not clip_path or clip_counts.get(clip_path, 0) > 1:
                continue
            target = Path(CLIPS_DIR / clip_path).resolve()
            # Path-traversal guard: only ever unlink inside CLIPS_DIR.
            with contextlib.suppress(OSError, ValueError):
                if target.is_file() and target.is_relative_to(clips_root):
                    target.unlink()
                    removed_clips += 1

    stats: dict[str, Any] = {
        "disabled": False,
        "events": len(expired_ids),
        "clips": removed_clips,
        "retention_days": RETENTION_DAYS,
    }
    logger.info("Retention sweep: %s", stats)
    return stats


async def _loop() -> None:
    while True:
        try:
            await purge_expired()
        except Exception:  # never kill the worker loop
            logger.exception("Retention sweep failed")
        await asyncio.sleep(max(RETENTION_SWEEP_HOURS, 0.1) * 3600)


def start_retention_task() -> asyncio.Task | None:
    """Start the background retention loop. No-op when purging is disabled."""
    if RETENTION_DAYS <= 0:
        logger.info("Retention purging disabled (RETENTION_DAYS=0)")
        return None
    return asyncio.create_task(_loop(), name="argus-retention")
