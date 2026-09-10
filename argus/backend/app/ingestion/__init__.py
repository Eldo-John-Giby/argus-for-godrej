"""Video ingestion — file upload, frame extraction, RTSP stub."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional


class VideoIngestion:
    """Handle video file upload and frame extraction."""

    def __init__(self, upload_dir: str = "./storage/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> Path:
        """Save an uploaded video file."""
        ext = Path(filename).suffix or ".mp4"
        safe_name = f"{uuid.uuid4().hex[:12]}{ext}"
        path = self.upload_dir / safe_name
        path.write_bytes(content)
        return path

    def extract_frames(
        self,
        video_path: str | Path,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Extract frame indices and timestamps from a video.

        Returns list of (frame_number, timestamp_seconds).
        Does not decode pixels — just reads metadata.
        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(source_fps / fps) if fps > 0 else 1

            frames = []
            for i in range(0, total_frames, frame_interval):
                timestamp = i / source_fps
                frames.append((i, timestamp))
                if max_frames and len(frames) >= max_frames:
                    break

            cap.release()
            return frames

        except ImportError:
            return []

    def get_video_metadata(self, video_path: str | Path) -> dict:
        """Get video metadata without reading frames."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {}

            metadata = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            }
            cap.release()
            return metadata

        except ImportError:
            return {}
