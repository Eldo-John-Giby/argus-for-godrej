"""ARGUS — AI Field Intelligence for Warehouse Handling.

FastAPI main application. Entry point for the backend server.
"""

import contextlib
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import STORAGE_ROOT
from app.database import engine, Base
from app.api import events, feedback, assistant, health, ingestion, bays, reports, alerts
from app.retention import start_retention_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — create tables, start retention worker."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    retention_task = start_retention_task()
    try:
        yield
    finally:
        if retention_task is not None:
            retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await retention_task
        await engine.dispose()


app = FastAPI(
    title="ARGUS",
    description="AI Field Intelligence for Warehouse Handling — Godrej Enterprises Group × graVITas'2026",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(events.router)
app.include_router(feedback.router)
app.include_router(assistant.router)
app.include_router(ingestion.router)
app.include_router(bays.router)
app.include_router(reports.router)
app.include_router(alerts.router)

# Serve event evidence clips statically. The seeder copies demo clips from
# ml/clips/<video_slug>/ into STORAGE_ROOT/clips; rows store paths relative
# to the clips root and the API maps them to /clips/<...> URLs.
# resolve() keeps the mount path absolute so StaticFiles can validate.
_CLIPS_DIR = (STORAGE_ROOT / "clips").resolve()
_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(_CLIPS_DIR)), name="clips")


@app.get("/")
async def root():
    return {
        "name": "ARGUS",
        "tagline": "Argus never blinks — an AI that watches every load so nothing gets dropped.",
        "version": "0.1.0",
        "docs": "/docs",
    }
