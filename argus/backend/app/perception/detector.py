"""YOLO detection + ByteTrack tracking wrapper.

Wraps Ultralytics YOLO for detect+pose+track, producing persistent
object IDs via ByteTrack. This is the main perception entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """A single detected object with bounding box, class, confidence, and optional keypoints."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    keypoints: Optional[dict] = None  # pose keypoints if available


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    frame_shape: tuple[int, int] = (0, 0)  # height, width


class YOLODetector:
    """Wrapper around Ultralytics YOLO for detection + tracking.

    In production, this would use YOLO26 with ByteTrack.
    For the hackathon scaffold, we provide a clean interface that
    can be swapped with the real model.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.7,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except ImportError:
                # Fallback: return empty detections for scaffold mode
                self._model = "scaffold"

    def detect_and_track(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> FrameDetections:
        """Run detection + tracking on a single frame.

        Returns FrameDetections with persistent track IDs via ByteTrack.
        """
        if timestamp is None:
            timestamp = time.time()

        self._load_model()

        if self._model == "scaffold":
            return FrameDetections(
                timestamp=timestamp,
                detections=[],
                frame_shape=(frame.shape[0], frame.shape[1]),
            )

        # Real YOLO + ByteTrack inference
        results = self._model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            tracker="bytetrack.yaml",
            device=self.device,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
                    track_id = int(box.id[0]) if box.id is not None else None

                    # Extract pose keypoints if available
                    kpts = None
                    if result.keypoints is not None and track_id is not None:
                        kpts_data = result.keypoints.xy[0].cpu().numpy()
                        kpts = {f"kp_{i}": (float(kpts_data[i][0]), float(kpts_data[i][1]))
                                for i in range(len(kpts_data))
                                if kpts_data[i][0] > 0 and kpts_data[i][1] > 0}

                    detections.append(Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_name=cls_name,
                        confidence=conf,
                        track_id=track_id,
                        keypoints=kpts,
                    ))

        return FrameDetections(
            timestamp=timestamp,
            detections=detections,
            frame_shape=(frame.shape[0], frame.shape[1]),
        )
