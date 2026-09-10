"""Clip extractor — extracts ±2s video clips around candidate events.

Uses OpenCV for frame-accurate extraction from video files.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from app.config import CLIPS_DIR, KEYFRAMES_DIR, ensure_storage_dirs


class ClipExtractor:
    """Extract video clips and keyframes around candidate events."""

    def __init__(
        self,
        clip_duration_before: float = 2.0,
        clip_duration_after: float = 2.0,
    ):
        self.clip_duration_before = clip_duration_before
        self.clip_duration_after = clip_duration_after

    def extract_clip(
        self,
        video_path: str,
        event_timestamp: float,
        fps: float = 30.0,
    ) -> Optional[Path]:
        """Extract a ±N second clip around the event timestamp.

        Args:
            video_path: Path to the source video file.
            event_timestamp: Timestamp of the event in seconds.
            fps: Frame rate of the source video.

        Returns:
            Path to the extracted clip, or None on failure.
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            event_frame = int(event_timestamp * source_fps)
            start_frame = max(0, event_frame - int(self.clip_duration_before * source_fps))
            end_frame = min(total_frames, event_frame + int(self.clip_duration_after * source_fps))

            clip_id = uuid.uuid4().hex[:12]
            ensure_storage_dirs()
            clip_path = CLIPS_DIR / f"clip_{clip_id}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(clip_path), fourcc, source_fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            cap.release()
            writer.release()

            return clip_path

        except ImportError:
            # OpenCV not available — skip clip extraction
            return None
        except Exception:
            return None

    def extract_keyframe(
        self,
        video_path: str,
        event_timestamp: float,
        fps: float = 30.0,
    ) -> Optional[Path]:
        """Extract a single keyframe at the event timestamp.

        Returns:
            Path to the extracted keyframe image, or None on failure.
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            event_frame = int(event_timestamp * source_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, event_frame)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            keyframe_id = uuid.uuid4().hex[:12]
            ensure_storage_dirs()
            keyframe_path = KEYFRAMES_DIR / f"keyframe_{keyframe_id}.jpg"
            cv2.imwrite(str(keyframe_path), frame)

            return keyframe_path

        except ImportError:
            return None
        except Exception:
            return None
