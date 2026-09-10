"""End-to-end perception pipeline test.

Tests: YOLO detection → ByteTrack tracking → FSM behaviour detection →
geometry features → risk scoring → confidence gate.

Generates a synthetic test video with moving boxes and runs the full pipeline.
"""

import os
import sys
import time
import tempfile
import json

import cv2
import numpy as np


def create_synthetic_video(output_path: str, duration_sec: float = 5.0, fps: float = 30.0) -> dict:
    """Create a synthetic warehouse video with moving objects for testing.

    Simulates:
    - A box being carried across the frame
    - A box being dropped (sudden downward motion)
    - A person dragging a box along the floor

    Returns metadata about what was placed in the video.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    metadata = {"objects": [], "events": []}

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # Dark background (warehouse floor)

        # Draw grid lines for floor reference
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (60, 60, 60), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (60, 60, 60), 1)

        # Object 1: Box being carried (moves left to right, then drops at t=3s)
        if t < 4.0:
            if t < 3.0:
                # Carrying: steady horizontal motion
                box_x = int(50 + (t / 3.0) * 400)
                box_y = int(200 + 10 * np.sin(t * 2))  # slight wobble
            else:
                # Drop: sudden downward motion after t=3s
                drop_t = t - 3.0
                box_x = int(450)
                box_y = int(200 + 0.5 * 980 * drop_t ** 2)  # gravity
                box_y = min(box_y, 420)  # floor

            box_w, box_h = 60, 50
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 165, 255), 2)
            cv2.putText(frame, "BOX", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            metadata["objects"].append({
                "type": "box",
                "frame": frame_idx,
                "bbox": [box_x, box_y, box_x + box_w, box_y + box_h],
                "event": "drop" if t >= 3.0 else "carry",
            })

        # Object 2: Box being dragged (moves along the floor)
        if 1.0 <= t <= 4.5:
            drag_x = int(100 + ((t - 1.0) / 3.5) * 300)
            drag_y = 420  # On the floor
            drag_w, drag_h = 50, 40
            cv2.rectangle(frame, (drag_x, drag_y - drag_h), (drag_x + drag_w, drag_y), (0, 200, 0), 2)
            cv2.putText(frame, "DRAG", (drag_x + 5, drag_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            metadata["objects"].append({
                "type": "box",
                "frame": frame_idx,
                "bbox": [drag_x, drag_y - drag_h, drag_x + drag_w, drag_y],
                "event": "drag",
            })

        # Object 3: Person (stick figure)
        if 0.5 <= t <= 4.0:
            person_x = int(80 + ((t - 0.5) / 3.5) * 350)
            person_y = 180
            # Head
            cv2.circle(frame, (person_x, person_y - 40), 12, (200, 200, 200), 2)
            # Body
            cv2.line(frame, (person_x, person_y - 28), (person_x, person_y + 30), (200, 200, 200), 2)
            # Arms
            cv2.line(frame, (person_x - 20, person_y), (person_x + 20, person_y), (200, 200, 200), 2)
            # Legs
            cv2.line(frame, (person_x, person_y + 30), (person_x - 15, person_y + 60), (200, 200, 200), 2)
            cv2.line(frame, (person_x, person_y + 30), (person_x + 15, person_y + 60), (200, 200, 200), 2)

            metadata["objects"].append({
                "type": "person",
                "frame": frame_idx,
                "bbox": [person_x - 20, person_y - 52, person_x + 20, person_y + 60],
                "event": "carry" if t < 3.0 else "drop_observation",
            })

        # Add some noise to make it more realistic
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        writer.write(frame)

    writer.release()
    print(f"  Created synthetic video: {output_path}")
    print(f"  Duration: {duration_sec}s, FPS: {fps}, Frames: {total_frames}")
    print(f"  Objects placed: {len(metadata['objects'])} frame entries")
    return metadata


def test_yolo_detection(video_path: str) -> dict:
    """Test YOLO detection + ByteTrack tracking on the video."""
    from ultralytics import YOLO

    print("\n--- YOLO Detection + Tracking ---")
    model = YOLO("yolov8n.pt")
    print(f"  Model loaded: yolov8n.pt")
    print(f"  Classes: {list(model.names.values())[:10]}...")

    results = model.track(
        video_path,
        persist=True,
        conf=0.25,
        iou=0.7,
        tracker="bytetrack.yaml",
        stream=True,
    )

    all_detections = []
    track_ids_seen = set()

    for frame_idx, result in enumerate(results):
        frame_detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                track_id = int(box.id[0]) if box.id is not None else None

                frame_detections.append({
                    "frame": frame_idx,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "track_id": track_id,
                })

                if track_id is not None:
                    track_ids_seen.add(track_id)

        all_detections.extend(frame_detections)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: {len(frame_detections)} detections, "
                  f"{len(track_ids_seen)} unique tracks")

    print(f"\n  Total detections: {len(all_detections)}")
    print(f"  Unique track IDs: {len(track_ids_seen)}")
    if all_detections:
        classes_found = set(d["class"] for d in all_detections)
        print(f"  Classes detected: {classes_found}")
        avg_conf = np.mean([d["confidence"] for d in all_detections])
        print(f"  Average confidence: {avg_conf:.3f}")

    return {
        "total_detections": len(all_detections),
        "unique_tracks": len(track_ids_seen),
        "classes": list(set(d["class"] for d in all_detections)),
        "avg_confidence": float(np.mean([d["confidence"] for d in all_detections])) if all_detections else 0,
        "detections": all_detections[:50],  # first 50 for inspection
    }


def test_perception_module(video_path: str):
    """Test the Argus perception module wrapper."""
    from app.perception.detector import YOLODetector

    print("\n--- Argus Perception Module ---")
    detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.25)

    # Read frames manually to test frame-by-frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    all_detections = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        frame_dets = detector.detect_and_track(frame, timestamp=timestamp)
        all_detections.append(frame_dets)

        if frame_count % 30 == 0:
            n_dets = len(frame_dets.detections)
            track_ids = [d.track_id for d in frame_dets.detections if d.track_id is not None]
            print(f"  Frame {frame_count} (t={timestamp:.2f}s): {n_dets} detections, "
                  f"tracks: {track_ids}")

        frame_count += 1

    cap.release()

    total_dets = sum(len(fd.detections) for fd in all_detections)
    print(f"\n  Frames processed: {frame_count}")
    print(f"  Total detections: {total_dets}")
    return all_detections


def test_fsm_with_detections(all_detections):
    """Test the behaviour FSM with real YOLO detections."""
    from app.behaviour.fsm import TrackFSM, FrameData, TrackState

    print("\n--- Behaviour FSM ---")

    # Group detections by track_id
    tracks: dict[int, list] = {}
    for fd in all_detections:
        for det in fd.detections:
            if det.track_id is not None:
                if det.track_id not in tracks:
                    tracks[det.track_id] = []
                tracks[det.track_id].append({
                    "timestamp": fd.timestamp,
                    "bbox": det.bbox,
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "keypoints": det.keypoints,
                })

    print(f"  Unique tracks to process: {len(tracks)}")

    fsm_results = {}
    for track_id, detections in tracks.items():
        fsm = TrackFSM(track_id=track_id, object_class=detections[0]["class"])
        events = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            frame_data = FrameData(
                timestamp=det["timestamp"],
                bbox=tuple(det["bbox"]),
                center=center,
                bottom_y=y2,
                area=(x2 - x1) * (y2 - y1),
                keypoints=det.get("keypoints"),
            )
            event = fsm.update(frame_data)
            if event:
                events.append({
                    "type": event.event_type.value,
                    "timestamp": event.timestamp,
                    "confidence": round(event.confidence, 3),
                    "features": event.features,
                })

        fsm_results[track_id] = {
            "class": detections[0]["class"],
            "num_frames": len(detections),
            "final_state": fsm.state.value,
            "events": events,
        }

        if events:
            print(f"  Track {track_id} ({detections[0]['class']}): "
                  f"{len(events)} events detected")
            for e in events:
                print(f"    → {e['type']} at t={e['timestamp']:.2f}s "
                      f"(confidence: {e['confidence']})")

    return fsm_results


def test_risk_scoring(fsm_results):
    """Test risk scoring on detected events."""
    from app.risk.scoring import RiskInput, score_event
    from app.risk.confidence_gate import ConfidenceInput, apply_confidence_gate

    print("\n--- Risk Scoring ---")

    all_scored = []
    for track_id, data in fsm_results.items():
        for event in data["events"]:
            risk_input = RiskInput(
                behaviour_severity=80 if "drop" in event["type"] or "throw" in event["type"] else 50,
                velocity_at_contact=event["features"].get("downward_velocity", 50),
                mass_proxy=1.0,
                fragility_multiplier=1.0,
                stack_tilt_angle=0,
                top_bottom_footprint_ratio=1.0,
                recurrence_count=1,
                composite_confidence=event["confidence"],
            )
            result = score_event(risk_input)

            # Apply confidence gate
            gate_input = ConfidenceInput(
                detector_confidence=event["confidence"],
                track_stability=min(event["confidence"] * 1.1, 1.0),
                vlm_agreement=-1,  # not called yet
            )
            gate_result = apply_confidence_gate(gate_input)

            scored = {
                "track_id": track_id,
                "event_type": event["type"],
                "risk_score": round(result.score, 1),
                "tier": result.tier,
                "status": gate_result.status,
                "should_alert": gate_result.should_alert,
                "composite_confidence": round(gate_result.composite_confidence, 3),
                "breakdown": {k: round(v, 1) for k, v in result.breakdown.items()},
            }
            all_scored.append(scored)

            print(f"  Track {track_id} — {event['type']}:")
            print(f"    Risk Score: {result.score:.1f}/100 ({result.tier})")
            print(f"    Confidence Gate: {gate_result.status} "
                  f"(composite: {gate_result.composite_confidence:.3f})")
            print(f"    Alert: {gate_result.should_alert}")
            print(f"    Breakdown: {scored['breakdown']}")

    return all_scored


def main():
    print("=" * 60)
    print("ARGUS Perception Pipeline — End-to-End Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_warehouse.mp4")

        # Step 1: Create synthetic test video
        print("\n[1/5] Creating synthetic warehouse video...")
        metadata = create_synthetic_video(video_path, duration_sec=3.0, fps=30)

        # Step 2: Run YOLO detection
        print("\n[2/5] Running YOLO detection + ByteTrack tracking...")
        yolo_results = test_yolo_detection(video_path)

        # Step 3: Test Argus perception module
        print("\n[3/5] Testing Argus perception module wrapper...")
        all_detections = test_perception_module(video_path)

        # Step 4: Run FSM
        print("\n[4/5] Running behaviour FSM on detections...")
        fsm_results = test_fsm_with_detections(all_detections)

        # Step 5: Risk scoring
        print("\n[5/5] Computing risk scores + confidence gate...")
        scored_events = test_risk_scoring(fsm_results)

        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Video: 3s synthetic warehouse scene")
        print(f"  YOLO detections: {yolo_results['total_detections']}")
        print(f"  Unique tracks: {yolo_results['unique_tracks']}")
        print(f"  Classes found: {yolo_results['classes']}")
        print(f"  Avg confidence: {yolo_results['avg_confidence']:.3f}")
        print(f"  FSM events detected: {sum(len(d['events']) for d in fsm_results.values())}")
        print(f"  Risk-scored events: {len(scored_events)}")
        if scored_events:
            tiers = {}
            for e in scored_events:
                tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
            print(f"  Tier distribution: {tiers}")
        print("=" * 60)
        print("✅ Pipeline test complete!")


if __name__ == "__main__":
    main()
