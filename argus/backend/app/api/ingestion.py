"""Video upload and ingestion API routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.ingestion import VideoIngestion

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])

_ingestion = VideoIngestion()


class IngestionResponse(BaseModel):
    video_id: str
    filename: str
    path: str
    metadata: dict


class FrameExtractionRequest(BaseModel):
    video_path: str
    fps: float = 1.0
    max_frames: int | None = None


@router.post("/upload", response_model=IngestionResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing.

    Uses a deterministic video_id based on filename + size so re-uploads
    of the same file return the same ID.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    saved_path = _ingestion.save_upload(file.filename, content)
    metadata = _ingestion.get_video_metadata(saved_path)

    # Deterministic ID: hash of filename + content size
    import hashlib
    video_id = hashlib.md5(f"{file.filename}:{len(content)}".encode()).hexdigest()[:12]

    return IngestionResponse(
        video_id=video_id,
        filename=file.filename,
        path=str(saved_path),
        metadata=metadata,
    )


@router.get("/frames")
async def extract_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int | None = None,
):
    """Extract frame indices and timestamps from a video."""
    frames = _ingestion.extract_frames(video_path, fps=fps, max_frames=max_frames)
    return {
        "video_path": video_path,
        "frame_count": len(frames),
        "frames": [{"frame_number": f[0], "timestamp": f[1]} for f in frames],
    }


@router.get("/metadata")
async def get_video_metadata(video_path: str):
    """Get video metadata (fps, dimensions, duration)."""
    metadata = _ingestion.get_video_metadata(video_path)
    if not metadata:
        raise HTTPException(status_code=404, detail="Video not found or unreadable")
    return metadata
