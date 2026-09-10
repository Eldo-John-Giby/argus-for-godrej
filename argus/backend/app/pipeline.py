"""Pipeline orchestrator — connects perception → FSM → risk → events.

This is the central processing pipeline that turns raw video frames
into persisted, scored, verified events in the database.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.behaviour.fsm import CandidateEvent, EventType, FrameData, TrackFSM, TrackState
from app.behaviour.features import extract_features, GeometryFeatures
from app.config import RISK_WEIGHTS
from app.perception.detector import Detection, FrameDetections, YOLODetector
from app.risk.confidence_gate import ConfidenceInput, apply_confidence_gate, compute_composite_confidence
from app.risk.scoring import RiskInput, score_event


@dataclass
class PipelineEvent:
    """A fully processed event ready for persistence."""
    event_type: EventType
    behaviour_type: str
    timestamp: float
    risk_score: float
    tier: str
    confidence: float
    status: str  # observed / potential / confirmed
    features: dict = field(default_factory=dict)
    risk_breakdown: dict = field(default_factory=dict)
    track_id: int = 0


class ProcessingPipeline:
    """End-to-end processing pipeline.

    Usage:
        pipeline = ProcessingPipeline(detector=YOLODetector())
        frame_events = pipeline.process_frame(frame, camera_id="cam_1")
        for event in frame_events:
            # persist to DB
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        fps: float = 30.0,
    ):
        self.detector = detector or YOLODetector()
        self.fps = fps

        # Per-track FSMs, keyed by ByteTrack ID
        self._track_fsms: dict[int, TrackFSM] = {}
        self._frame_count: int = 0
        self._prev_detections: dict[int, FrameData] = {}

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        camera_id: str = "default",
        bay_id: str = "default",
        exclude_classes: Optional[set[str]] = None,
    ) -> list[PipelineEvent]:
        """Process a single video frame through the full pipeline.

        Returns list of candidate events detected in this frame.
        "exclude_classes" skips detections of the given class names
        (e.g. persons) before the behaviour FSM, so operator motion
        does not masquerade as product handling.
        """
        if timestamp is None:
            timestamp = self._frame_count / self.fps
        self._frame_count += 1

        # Step 1: Perception — YOLO detect + track
        detections = self.detector.detect_and_track(frame, timestamp=timestamp)

        # Step 2: Build FrameData for each tracked object
        active_track_ids = set()
        events: list[PipelineEvent] = []
        for det in detections.detections:
            if det.track_id is None:
                continue
            if exclude_classes and det.class_name in exclude_classes:
                continue

            track_id = det.track_id
            active_track_ids.add(track_id)

            bbox = det.bbox
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bottom_y = y2
            area = (x2 - x1) * (y2 - y1)

            frame_data = FrameData(
                timestamp=timestamp,
                bbox=bbox,
                center=center,
                bottom_y=bottom_y,
                area=area,
                keypoints=det.keypoints,
                person_bottom_ys=tuple(
                    d.bbox[3] for d in detections.detections
                    if d.class_name == "person"
                ),
            )

            # Get or create FSM for this track
            if track_id not in self._track_fsms:
                self._track_fsms[track_id] = TrackFSM(
                    track_id=track_id,
                    object_class=det.class_name,
                )

            fsm = self._track_fsms[track_id]
            fsm._last_seen_frame = self._frame_count

            # Step 3: FSM state transition
            candidate = fsm.update(frame_data)

            # Store previous frame data for feature computation next frame
            self._prev_detections[track_id] = frame_data

            if candidate is not None:
                # Step 4: Compute geometry features
                geom_features = self._compute_features(
                    candidate, frame_data, fsm
                )

                # Step 5: Risk scoring
                risk_result = self._score_risk(candidate, geom_features, bay_id)

                # Step 6: Confidence gate
                confidence_input = ConfidenceInput(
                    detector_confidence=det.confidence,
                    track_stability=min(len(fsm.frame_history) / 30, 1.0),
                    vlm_agreement=-1,  # VLM not yet called
                )
                gate = apply_confidence_gate(confidence_input)

                events.append(PipelineEvent(
                    event_type=candidate.event_type,
                    behaviour_type=candidate.event_type.value,
                    timestamp=timestamp,
                    risk_score=risk_result.score,
                    tier=risk_result.tier,
                    confidence=gate.composite_confidence,
                    status=gate.status,
                    features=candidate.features,
                    risk_breakdown=risk_result.breakdown,
                    track_id=track_id,
                ))

        # Clean up FSMs for tracks that are no longer visible
        stale_ids = set(self._track_fsms.keys()) - active_track_ids
        for tid in stale_ids:
            fsm = self._track_fsms[tid]
            frames_since_active = self._frame_count - getattr(fsm, "_last_seen_frame", self._frame_count)
            # Remove if track has been gone for >60 frames (~2s at 30fps)
            if frames_since_active > 60:
                del self._track_fsms[tid]
                self._prev_detections.pop(tid, None)

        return events

    def _compute_features(
        self,
        candidate: CandidateEvent,
        frame_data: FrameData,
        fsm: TrackFSM,
    ) -> GeometryFeatures:
        """Compute geometry features from candidate event data."""
        prev_data = self._prev_detections.get(fsm.track_id)

        prev_bbox = prev_data.bbox if prev_data else None
        prev_velocity = prev_data.velocity if prev_data and hasattr(prev_data, 'velocity') else None

        return extract_features(
            bbox=frame_data.bbox,
            prev_bbox=prev_bbox,
            dt=1.0 / self.fps,
        )

    def _score_risk(
        self,
        candidate: CandidateEvent,
        geom_features: GeometryFeatures,
        bay_id: str,
    ):
        """Score the risk of a candidate event."""
        from app.behaviour.taxonomy import load_taxonomy, get_fragility

        taxonomy = load_taxonomy()
        behaviour_key = candidate.event_type.value
        severity_base = taxonomy.get(behaviour_key, {}).get("severity_base", 50)

        # Product-specific risk (PS): fragility multiplier from the tracked
        # object's class (taxonomy.yaml fragility tiers), and the per-bay
        # hazard factor as the location input.
        fsm = self._track_fsms.get(candidate.track_id)
        object_class = fsm.object_class if fsm is not None else ""

        risk_input = RiskInput(
            behaviour_severity=severity_base,
            velocity_at_contact=geom_features.velocity_magnitude,
            mass_proxy=candidate.features.get("mass_proxy", 1.0),
            fragility_multiplier=get_fragility(object_class),
            stack_tilt_angle=geom_features.tilt_angle,
            top_bottom_footprint_ratio=geom_features.footprint_ratio,
            recurrence_count=1,  # would need DB lookup
            composite_confidence=candidate.confidence,
            bay_id=bay_id,
        )

        return score_event(risk_input)

    async def verify_with_vlm(
        self,
        pipeline_event: PipelineEvent,
        clip_path: Optional[str] = None,
    ) -> PipelineEvent:
        """Run VLM verification on a candidate event and update confidence.

        Call this after clip extraction. If VLM confirms, confidence goes up;
        if VLM rejects, the event is demoted or dismissed.
        """
        if clip_path is None:
            return pipeline_event

        from pathlib import Path
        from app.vlm.verifier import VLMVerifier

        verifier = VLMVerifier()
        result = await verifier.verify_event(
            clip_path=Path(clip_path),
            behaviour_type=pipeline_event.behaviour_type,
            features=pipeline_event.features,
        )

        # Update event with VLM verification
        vlm_conf = result.confidence if result.verified else result.confidence * 0.3
        pipeline_event.confidence = min(
            pipeline_event.confidence * 0.6 + vlm_conf * 0.4,
            1.0,
        )
        pipeline_event.features["vlm_explanation"] = result.explanation
        pipeline_event.features["vlm_verified"] = result.verified

        # Re-run confidence gate with VLM signal
        from app.risk.confidence_gate import apply_confidence_gate, ConfidenceInput
        gate = apply_confidence_gate(ConfidenceInput(
            detector_confidence=pipeline_event.confidence,
            track_stability=0.8,
            vlm_agreement=vlm_conf,
        ))
        pipeline_event.status = gate.status

        return pipeline_event

    def get_track_count(self) -> int:
        """Number of active tracked objects."""
        return len(self._track_fsms)

    def reset(self):
        """Reset all pipeline state."""
        self._track_fsms.clear()
        self._prev_detections.clear()
        self._frame_count = 0
