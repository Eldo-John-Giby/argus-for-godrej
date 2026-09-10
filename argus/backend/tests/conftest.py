"""Integration test fixtures — in-memory SQLite, FastAPI TestClient, seed data."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models import Camera, Event, Feedback, Track

# ---------------------------------------------------------------------------
# In-memory async SQLite engine (one per test module via function scope)
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture()
async def db_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# FastAPI AsyncClient wired to the in-memory DB
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient that hits the real FastAPI app but uses the test DB."""
    from main import app  # import AFTER models are registered

    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Seed-data factories
# ---------------------------------------------------------------------------


async def seed_camera(session: AsyncSession, bay_id: str = "bay_A", name: str = "cam_1") -> Camera:
    cam = Camera(bay_id=bay_id, name=name)
    session.add(cam)
    await session.commit()
    await session.refresh(cam)
    return cam


async def seed_track(session: AsyncSession, camera_id: int, object_class: str = "person") -> Track:
    track = Track(
        camera_id=camera_id,
        object_class=object_class,
        byte_track_id=1,
        start_ts=dt.datetime.utcnow(),
    )
    session.add(track)
    await session.commit()
    await session.refresh(track)
    return track


async def seed_event(
    session: AsyncSession,
    track_id: int,
    behaviour_type: str = "3_product_thrown",
    risk_score: float = 87.2,
    tier: str = "Critical",
    confidence: float = 0.92,
    status: str = "potential",
    risk_breakdown: dict | None = None,
) -> Event:
    features = {"throw_velocity": [180, -220]}
    if risk_breakdown:
        features["risk_breakdown"] = risk_breakdown

    event = Event(
        track_id=track_id,
        behaviour_type=behaviour_type,
        ts=dt.datetime.utcnow(),
        features_json=features,
        risk_score=risk_score,
        tier=tier,
        confidence=confidence,
        status=status,
        vlm_explanation="Operator threw package from loading dock.",
    )
    session.add(event)
    await session.commit()
    await session.refresh(event)
    return event


async def seed_feedback(
    session: AsyncSession,
    event_id: int,
    action: str = "confirm",
    reviewer: str = "supervisor",
) -> Feedback:
    fb = Feedback(
        event_id=event_id,
        reviewer=reviewer,
        action=action,
        ts=dt.datetime.utcnow(),
    )
    session.add(fb)
    await session.commit()
    await session.refresh(fb)
    return fb


# ---------------------------------------------------------------------------
# Full seed: camera + track + multiple events + feedback
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def seeded_data(db_session: AsyncSession):
    """Seed two bays with cameras, tracks, events, and feedback."""
    cam_a = await seed_camera(db_session, bay_id="bay_A", name="cam_A1")
    cam_b = await seed_camera(db_session, bay_id="bay_B", name="cam_B1")

    track_a = await seed_track(db_session, camera_id=cam_a.id, object_class="person")
    track_b = await seed_track(db_session, camera_id=cam_b.id, object_class="person")

    events = []
    # Bay A: 3 events (2 Critical, 1 High)
    for behaviour, score, tier, status in [
        ("3_product_thrown", 87.2, "Critical", "potential"),
        ("1_product_dropped", 68.5, "High", "observed"),
        ("3_product_thrown", 91.0, "Critical", "potential"),
    ]:
        e = await seed_event(
            db_session,
            track_id=track_a.id,
            behaviour_type=behaviour,
            risk_score=score,
            tier=tier,
            status=status,
            risk_breakdown={
                "behaviour_severity": 90,
                "impact_proxy": 72.5,
                "fragility": 45.0,
                "stack_instability": 0,
                "recurrence": 15.0,
                "confidence_penalty": -3.2,
            },
        )
        events.append(e)

    # Bay B: 2 events (1 Medium, 1 Low)
    for behaviour, score, tier, status in [
        ("2_product_dragged", 45.3, "Medium", "dismissed"),
        ("5_unstable_stacking", 20.1, "Low", "observed"),
    ]:
        e = await seed_event(
            db_session,
            track_id=track_b.id,
            behaviour_type=behaviour,
            risk_score=score,
            tier=tier,
            status=status,
        )
        events.append(e)

    # Feedback on first 2 events
    await seed_feedback(db_session, event_id=events[0].id, action="confirm")
    await seed_feedback(db_session, event_id=events[1].id, action="dismiss")
    await seed_feedback(db_session, event_id=events[2].id, action="confirm")
    await seed_feedback(db_session, event_id=events[3].id, action="dismiss")

    return {
        "cameras": [cam_a, cam_b],
        "tracks": [track_a, track_b],
        "events": events,
    }
