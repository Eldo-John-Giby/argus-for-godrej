#!/usr/bin/env python
"""Run the full ARGUS pipeline over the judge-supplied warehouse videos.

For each video in ``argus/videos`` this script:

  1. Detects + tracks persons and products with YOLO-World (open-vocabulary,
     prompts tuned per scenario) using ByteTrack persistent IDs.
  2. Feeds non-person tracks through the behaviour FSM -> physics-informed
     risk scoring -> confidence gate (the real backend pipeline).
  3. Adds cross-track signals the per-track FSM cannot see:
     - 6_stepping_on_packages   (person feet/bbox over a carton)
     - 4_improper_stacking      (top carton footprint > bottom carton)
     - 11_wrong_orientation     (product lying horizontally when it should stand)
  4. Writes per-video JSON reports, extracts demo clips (mp4 + annotated JPGs),
     and emits a summary.json consumed by ml/eval.py for precision/recall.

Clip encoding: event clips are written as browser-playable H.264 via ffmpeg
(system PATH or the `imageio-ffmpeg` pip package — ML environment only, the
backend image does not need it). Without ffmpeg, clips fall back to mp4v
(desktop-only playback).

Usage:
    python run_on_videos.py                 # process all videos
    python run_on_videos.py --limit 1       # first video only (iteration)
    python run_on_videos.py --no-clips      # skip clip extraction
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.behaviour.fsm import EventType, FrameData, TrackFSM, TrackState  # noqa: E402
from app.behaviour.taxonomy import get_fragility, load_taxonomy  # noqa: E402
from app.perception.detector import Detection, FrameDetections  # noqa: E402
from app.pipeline import ProcessingPipeline  # noqa: E402
from face_blur import face_boxes_from_persons, maybe_blur_frame  # noqa: E402
from ppe_detector import PPEDetector  # noqa: E402
from app.risk.confidence_gate import ConfidenceInput, apply_confidence_gate  # noqa: E402
from app.risk.scoring import RiskInput, score_event  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "videos"
RESULTS_DIR = ROOT / "ml" / "results"
CLIPS_DIR = ROOT / "clips"

# Run the PPE detector every Nth frame (CPU budget); dedup absorbs the gaps.
PPE_STRIDE = 3

## Universal warehouse open-vocabulary classes.
## Applied uniformly across all loading bays, dock levels, and storage areas without per-video tailoring.
UNIVERSAL_WAREHOUSE_CLASSES = [
    "person", "cardboard box", "carton", "box", "wooden pallet",
    "pallet", "trolley", "mattress", "cabinet", "cupboard", "appliance", "goods",
]

SCENARIO_PROMPTS = {name: UNIVERSAL_WAREHOUSE_CLASSES for name in [
    "Dock level, dragging cupboard.mp4",
    "KD packets dragged, heavy box kept on other packets.mp4",
    "Rolling and dragging on wet floor.mp4",
    "Rolling and dropping carton.mp4",
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4",
    "Throwing Mattresses.mp4",
    "Throwing seating cartons, using strap to hold.mp4",
]}

# Ground truth behaviour labels, taken verbatim from the judge video names.
GROUND_TRUTH = {
    "Dock level, dragging cupboard.mp4": ["2_product_dragged", "8_dragging_no_trolley"],
    "KD packets dragged, heavy box kept on other packets.mp4": ["2_product_dragged", "4_improper_stacking"],
    "Rolling and dragging on wet floor.mp4": ["2_product_dragged"],
    "Rolling and dropping carton.mp4": ["1_product_dropped", "2_product_dragged"],
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4": [
        "6_stepping_on_packages", "11_wrong_orientation", "4_improper_stacking",
    ],
    "Throwing Mattresses.mp4": ["3_product_thrown"],
    "Throwing seating cartons, using strap to hold.mp4": ["3_product_thrown"],
}


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:60].strip("_")


class WorldDetector:
    """YOLO-World open-vocabulary detector + ByteTrack tracker.

    Mirrors the interface of ``YOLODetector.detect_and_track`` so the existing
    ProcessingPipeline consumes it unchanged. Detection runs at 640x360 for
    speed; boxes are scaled back to the original frame resolution so the
    behaviour FSM's 720p-tuned thresholds stay valid.
    """

    def __init__(self, classes, model_path="yolov8s-world.pt", conf=0.035, iou=0.7):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.set_classes(classes)
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.last = FrameDetections(timestamp=0.0)
        self._next_fallback_id = 500
        self._fallback_tracks = {}  # id -> (bbox, last_seen_time, class_name)

    def detect_and_track(self, frame, timestamp=None):
        t = timestamp or 0.0
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, 360))
        scale = w / 640.0
        tracker_cfg = str(Path(__file__).resolve().parent / "custom_bytetrack.yaml")
        if not Path(tracker_cfg).exists():
            tracker_cfg = "custom_bytetrack.yaml"
        results = self.model.track(
            small, persist=True, conf=self.conf, iou=self.iou,
            tracker=tracker_cfg, imgsz=384, device="cpu", verbose=False,
        )
        dets = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() * scale
                bw, bh = x2 - x1, y2 - y1
                # Guard against spurious full-frame / giant hallucinated boxes
                if bw > 0.90 * w and bh > 0.85 * h:
                    continue
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf_val = float(box.conf[0])
                bbox = (float(x1), float(y1), float(x2), float(y2))

                tid = int(box.id[0]) if box.id is not None else None
                if tid is None:
                    # Match to fallback tracks
                    best_id = None
                    best_iou = 0.15
                    for fid, (fb_box, fb_t, fb_cls) in list(self._fallback_tracks.items()):
                        if t - fb_t > 2.0:
                            del self._fallback_tracks[fid]
                            continue
                        if fb_cls == cls_name:
                            ix1 = max(bbox[0], fb_box[0])
                            iy1 = max(bbox[1], fb_box[1])
                            ix2 = min(bbox[2], fb_box[2])
                            iy2 = min(bbox[3], fb_box[3])
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                union = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) + (fb_box[2]-fb_box[0])*(fb_box[3]-fb_box[1]) - inter
                                iou = inter / union if union > 0 else 0
                                if iou > best_iou:
                                    best_iou = iou
                                    best_id = fid
                    if best_id is None:
                        self._next_fallback_id += 1
                        best_id = self._next_fallback_id
                    tid = best_id
                    self._fallback_tracks[tid] = (bbox, t, cls_name)
                else:
                    self._fallback_tracks[tid] = (bbox, t, cls_name)

                dets.append(Detection(
                    bbox=bbox,
                    class_name=cls_name,
                    confidence=conf_val,
                    track_id=tid,
                ))
        self.last = FrameDetections(timestamp=timestamp, detections=dets, frame_shape=(h, w))
        return self.last


class CrossTrackSignals:
    """Signals that need two tracks at once — invisible to the per-track FSM."""

    OBJECT_KEYWORDS = ("box", "carton", "packet", "package", "parcel", "cupboard",
                       "cabinet", "wardrobe", "furniture", "mattress", "chair", "sofa",
                       "appliance", "goods", "load", "drum", "crate")
    PACKAGE_CLASSES = {"box", "cardboard box", "carton", "packet", "package", "parcel", "bag", "goods"}
    CARGO_CLASSES = {
        "box", "cardboard box", "carton", "packet", "package", "parcel",
        "bag", "goods", "cabinet", "cupboard", "appliance", "furniture", "load", "crate"
    }

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        # (signal, track_id) -> last emission time
        self._last_emit: dict[tuple, float] = {}
        # (signal, track_id) -> running frame evidence
        self._evidence: dict[tuple, int] = {}
        self._step_consec: dict[int, int] = {}
        self._orient_consec: dict[int, int] = {}
        self._person_last: dict[int, tuple] = {}   # track_id -> (bottom_y, t)
        self._obj_last: dict[int, tuple] = {}      # track_id -> (bbox, t)
        self._obj_speed: dict[int, float] = {}     # track_id -> smoothed px/s
        self._obj_vy: dict[int, float] = {}        # track_id -> smoothed vertical px/s
        self._pair_frames: dict[tuple, int] = {}   # (top_id, bottom_id) -> stacked frames
        self._pair_since: dict[tuple, float] = {}  # (top_id, bottom_id) -> first stack t
        self._package_anchors: dict[int, tuple] = {} # track_id -> (bbox, t_last_seen, class_name)
        self.cooldown = 4.0

    def _emit(self, signal, track_id, timestamp, features, conf):
        key = (signal, track_id)
        if timestamp - self._last_emit.get(key, -1e9) < self.cooldown:
            return None
        self._last_emit[key] = timestamp
        severity = self.taxonomy.get(signal, {}).get("severity_base", 50)
        # Product-specific risk: fragility tier from the at-risk object's
        # class when the signal carries one (stepping -> on_class, stacking
        # -> top_class, orientation -> class); persons default to Standard.
        obj_class = (features.get("top_class") or features.get("on_class")
                     or features.get("class") or "")
        risk = score_event(RiskInput(
            behaviour_severity=severity,
            velocity_at_contact=features.get("velocity", 0.0),
            mass_proxy=1.0,
            fragility_multiplier=get_fragility(obj_class),
            stack_tilt_angle=0.0, top_bottom_footprint_ratio=features.get("footprint_ratio", 1.0),
            recurrence_count=1, composite_confidence=conf,
        ))
        gate = apply_confidence_gate(ConfidenceInput(
            detector_confidence=conf, track_stability=min(features.get("frames", 0) / 30, 1.0),
            vlm_agreement=-1,
        ))
        return {
            "behaviour_type": signal,
            "timestamp": timestamp,
            "risk_score": round(risk.score, 1),
            "tier": risk.tier,
            "confidence": round(gate.composite_confidence, 3),
            "status": gate.status,
            "features": features,
            "track_id": track_id,
            "source": "cross_track",
        }

    def update(self, frame_dets: FrameDetections) -> list[dict]:
        events = []
        t = frame_dets.timestamp or 0.0
        persons = [d for d in frame_dets.detections if d.class_name == "person" and d.track_id is not None]
        objects = [d for d in frame_dets.detections
                   if d.class_name != "person" and d.track_id is not None]

        # Update package memory anchor (for occlusion handling)
        for o in objects:
            if o.class_name in self.PACKAGE_CLASSES:
                self._package_anchors[o.track_id] = (o.bbox, t, o.class_name)

        # Cleanup anchors not seen for > 4.0 seconds
        for tid, (abox, at, acls) in list(self._package_anchors.items()):
            if t - at > 4.0:
                del self._package_anchors[tid]

        # Combine active objects + recently seen package anchors for stepping check
        candidate_packages = []
        seen_tids = set()
        for o in objects:
            if o.class_name in self.PACKAGE_CLASSES:
                candidate_packages.append((o.bbox, o.class_name, o.confidence))
                seen_tids.add(o.track_id)
        for tid, (abox, at, acls) in self._package_anchors.items():
            if tid not in seen_tids:
                candidate_packages.append((abox, acls, 0.5))

        # --- 6_stepping_on_packages: person actively standing or walking on a flat package ---
        active_stepping_pids = set()
        for p in persons:
            if p.confidence < 0.28:
                continue
            x1, y1, x2, y2 = p.bbox
            p_bottom = y2
            pw = x2 - x1
            pcx = (x1 + x2) / 2.0

            for (obbox, oclass, oconf) in candidate_packages:
                ox1, oy1, ox2, oy2 = obbox
                ow = ox2 - ox1
                oh = oy2 - oy1
                # Must be a flat floor package (oh <= 110, ow >= 35, bottom near floor)
                if ow < 35 or oh < 20 or oh > 110:
                    continue
                if oy2 < 440.0:
                    continue

                # Person center is horizontally over the package with substantial width overlap
                in_bounds_x = (ox1 - 10.0 <= pcx <= ox2 + 10.0)
                overlap_x = min(x2, ox2) - max(x1, ox1)
                if not (in_bounds_x and overlap_x >= 0.45 * pw):
                    continue

                # Feet must rest directly on the top surface of the package, elevated above the floor
                on_top = (oy1 - 12.0 <= p_bottom <= oy1 + 22.0) and (p_bottom <= oy2 - 15.0)
                if on_top:
                    active_stepping_pids.add(p.track_id)
                    frames_cnt = self._step_consec.get(p.track_id, 0) + 1
                    self._step_consec[p.track_id] = frames_cnt
                    if frames_cnt >= 12:
                        ev = self._emit("6_stepping_on_packages", p.track_id, t,
                                        {"frames": frames_cnt, "on_class": oclass,
                                         "person_bottom": round(p_bottom, 1), "velocity": 0.0,
                                         "footprint_ratio": 1.0}, p.confidence)
                        if ev:
                            ev["features"]["carton_bbox"] = [round(v, 1) for v in obbox]
                            events.append(ev)
                    break
            self._person_last[p.track_id] = (p_bottom, t)

        # Reset consecutive counter for persons not actively stepping
        for pid in list(self._step_consec.keys()):
            if pid not in active_stepping_pids:
                self._step_consec[pid] = max(0, self._step_consec[pid] - 2)

        # --- 4_improper_stacking ---
        # Stacking between cargo items (heavy box/cabinet on other packets, or moving base)
        active_stack_pairs = set()
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                if a.class_name not in self.CARGO_CLASSES or b.class_name not in self.CARGO_CLASSES:
                    continue
                top, bottom = (a, b) if a.bbox[1] < b.bbox[1] else (b, a)
                if abs(top.bbox[1] - bottom.bbox[1]) < 15:
                    continue
                tx1, ty1, tx2, ty2 = top.bbox
                bx1, by1, bx2, by2 = bottom.bbox
                # Skip ceiling/wall shelves (must be in operating floor space)
                if ty1 < 50.0 or by1 < 50.0:
                    continue
                tw, th = tx2 - tx1, ty2 - ty1
                bw, bh = bx2 - bx1, by2 - by1
                stack_band = (by1 - 45 <= ty2 <= by1 + 0.75 * bh)
                overlap_x = min(tx2, bx2) - max(tx1, bx1)

                # Check vertical motion: neither object should be falling/flying through air
                top_vy = self._obj_vy.get(top.track_id, 0.0)
                if abs(top_vy) > 60.0:
                    continue

                if stack_band and overlap_x > 0.25 * min(tw, bw):
                    top_area = tw * th
                    bottom_area = bw * bh
                    ratio = top_area / bottom_area if bottom_area > 0 else 0.0
                    base_speed = self._obj_speed.get(bottom.track_id, 0.0)
                    pair = tuple(sorted((top.track_id, bottom.track_id)))
                    active_stack_pairs.add(pair)
                    if pair not in self._pair_since:
                        self._pair_since[pair] = t
                    self._pair_frames[pair] = self._pair_frames.get(pair, 0) + 1

                    # Improper if:
                    # 1) Heavy product (cupboard/cabinet/appliance/furniture) kept on top of cargo
                    # 2) Disproportionate large item on smaller base (ratio >= 1.05)
                    # 3) Stacking on moving base under transport (base_speed >= 25.0)
                    is_heavy_top = top.class_name in {"cupboard", "cabinet", "appliance", "furniture"}
                    improper = (is_heavy_top or ratio >= 1.05 or base_speed >= 25.0)
                    sustained = (self._pair_frames[pair] >= 4 and t - self._pair_since[pair] >= 0.12)
                    if sustained and improper:
                        ev = self._emit("4_improper_stacking", top.track_id, t,
                                        {"frames": self._bump("4", top.track_id),
                                         "top_class": top.class_name, "bottom_class": bottom.class_name,
                                         "footprint_ratio": round(ratio, 2),
                                         "base_speed": round(base_speed, 1),
                                         "velocity": base_speed},
                                        max(top.confidence, bottom.confidence))
                        if ev:
                            ev["features"]["top_bbox"] = [round(v, 1) for v in top.bbox]
                            ev["features"]["bottom_bbox"] = [round(v, 1) for v in bottom.bbox]
                            events.append(ev)

        # Decay inactive pairs
        for pair in list(self._pair_frames.keys()):
            if pair not in active_stack_pairs:
                self._pair_frames[pair] = max(0, self._pair_frames[pair] - 1)

        # --- 11_wrong_orientation + per-object speed tracking ---
        for o in objects:
            ox1, oy1, ox2, oy2 = o.bbox
            w, h = ox2 - ox1, oy2 - oy1
            if h <= 0:
                continue
            aspect = w / h
            prev = self._obj_last.get(o.track_id)
            if prev and (t - prev[1]) > 0:
                dt_obj = t - prev[1]
                px, py = (prev[0][0] + prev[0][2]) / 2, (prev[0][1] + prev[0][3]) / 2
                cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                inst = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / dt_obj
                inst_vy = (cy - py) / dt_obj
                self._obj_speed[o.track_id] = 0.7 * self._obj_speed.get(o.track_id, 0.0) + 0.3 * inst
                self._obj_vy[o.track_id] = 0.7 * self._obj_vy.get(o.track_id, 0.0) + 0.3 * inst_vy

            # Wrong orientation: vertical product (tall item) kept flat horizontally
            # Intrinsically applies to upright products (cupboard, cabinet, appliance, furniture, wardrobe)
            # or elongated tall items lying horizontally (aspect >= 2.60 with width >= 220)
            is_upright_class = o.class_name in {"cupboard", "cabinet", "appliance", "furniture", "wardrobe"}
            is_wrong_orient = (
                (is_upright_class and aspect >= 1.40 and w >= 150)
                or (aspect >= 2.60 and w >= 220 and oy2 >= 450.0)
            )
            if is_wrong_orient:
                frames_cnt = self._orient_consec.get(o.track_id, 0) + 1
                self._orient_consec[o.track_id] = frames_cnt
                if frames_cnt >= 8:
                    ev = self._emit("11_wrong_orientation", o.track_id, t,
                                    {"frames": frames_cnt, "aspect_ratio": round(aspect, 2),
                                     "class": o.class_name, "velocity": 0.0,
                                     "footprint_ratio": round(aspect, 2)}, o.confidence)
                    if ev:
                        events.append(ev)
            else:
                self._orient_consec[o.track_id] = max(0, self._orient_consec.get(o.track_id, 0) - 1)
            self._obj_last[o.track_id] = (o.bbox, t)

        return events

    def _bump(self, signal, track_id):
        key = (signal, track_id)
        self._evidence[key] = self._evidence.get(key, 0) + 1
        return self._evidence[key]


def annotate(frame, dets, events_this_frame=None, scale=1.0, ppe_violations=None):
    """Draw detections (with class + track id) on a copy of the frame."""
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v * scale) for v in d.bbox]
        color = (255, 128, 0) if d.class_name == "person" else (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} #{d.track_id}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # PPE violations: red box on the person, legible in a 2-second clip
    for v in (ppe_violations or []):
        x1, y1, x2, y2 = [int(c * scale) for c in v.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = "NO-" + "/NO-".join(m.upper() for m in v.missing) + f" #{v.track_id}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if events_this_frame:
        for e in events_this_frame:
            cv2.putText(img, f"{e['behaviour_type']} risk={e['risk_score']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


def _ffmpeg_exe():
    """Locate an ffmpeg binary (system PATH first, then imageio-ffmpeg)."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_clip(video_path, out_mp4, start_t, end_t, fps):
    """Copy the raw source segment [start_t, end_t] into a small mp4 clip.

    Encodes H.264 (yuv420p + faststart) so clips play in Chrome/Edge/Safari —
    OpenCV's default mp4v/FMP4 output is NOT decodable by browsers, which
    silently renders as an unplayable player. Falls back to mp4v only when
    no ffmpeg binary can be found (local OpenCV playback still works there).
    """
    cap = cv2.VideoCapture(str(video_path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        # Pipe raw BGR frames into ffmpeg -> H.264 mp4.
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{vfps:.6f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_mp4),
        ]
        try:
            import subprocess

            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
            written = 0
            while written < int((end_t - start_t) * vfps):
                ok, frame = cap.read()
                if not ok:
                    break
                proc.stdin.write(frame.tobytes())
                written += 1
            cap.release()
            proc.stdin.close()
            ok_out = proc.wait(timeout=60) == 0
            if ok_out and written > 5:
                return True
            # ffmpeg failed oddly — clean partial output, use fallback below.
            Path(out_mp4).unlink(missing_ok=True)
        except Exception:
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            Path(out_mp4).unlink(missing_ok=True)

    # Fallback: OpenCV writer (mp4v — desktop-playable, not browser-playable).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, vfps, (w, h))
    if not writer.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    written = 0
    while written < int((end_t - start_t) * vfps):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    return written > 5


def annotate_and_encode_clip(video_path, out_mp4, start_t, end_t, fps,
                             detector, pipeline, ppe, blur_enabled, event=None):
    """Extract an event window with detections burned in (H.264).

    Re-runs the detector inside the ±1.5s window so every stored clip shows
    the boxes a judge expects: object/person tracks, red PPE violations, the
    behaviour banner, and face pixelation. Falls back to a raw extract when
    no ffmpeg binary is available.
    """
    cap = cv2.VideoCapture(str(video_path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        cap.release()
        return False

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{vfps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    written = 0
    try:
        while written < int((end_t - start_t) * vfps):
            ok, frame = cap.read()
            if not ok:
                break
            t = start_t + written / vfps
            pipeline.process_frame(
                frame, timestamp=t, camera_id="clip",
                exclude_classes={"person", "trolley", "pallet", "wooden pallet",
                                 "cart", "hand truck", "dolly"},
            )
            dets = detector.last.detections
            persons = [d for d in dets if d.class_name == "person"]
            violations = None
            if ppe is not None and ppe.available and persons:
                violations = ppe.check_persons(frame, persons)
            img = annotate(frame, dets, [event] if event else None,
                           ppe_violations=violations)
            if blur_enabled:
                img = maybe_blur_frame(img, face_boxes_from_persons(persons), blur_enabled)
            proc.stdin.write(img.tobytes())
            written += 1
        proc.stdin.close()
        ok_out = proc.wait(timeout=120) == 0 and written > 5
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        ok_out = False
    cap.release()
    if not ok_out:
        Path(out_mp4).unlink(missing_ok=True)
    return ok_out


def run_video(video_path: Path, args, detector_cls=WorldDetector) -> dict:
    name = video_path.name
    slug = slugify(name)
    out_dir = RESULTS_DIR / slug
    clip_dir = CLIPS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy()
    detector = detector_cls(SCENARIO_PROMPTS.get(
        name,
        # New/unknown video: use a general warehouse prompt so any dropped-in
        # file can be processed without editing this table first.
        "person, box, carton, package, trolley, pallet, wooden pallet, forklift, cupboard, mattress",
    ))
    pipeline = ProcessingPipeline(detector=detector)
    cross = CrossTrackSignals(taxonomy)

    # Responsible AI: face/ID blurring on stored frames (default ON)
    blur_enabled = not args.no_blur

    # Optional PPE compliance detector (second model, same frames)
    ppe = None
    if not args.no_ppe:
        ppe = PPEDetector(model_path=args.ppe_model)
        if ppe.available:
            print(f"  PPE detector loaded: {ppe.weight_path.name}")
        else:
            print("  PPE detector: no weight found (ml/ppe_best.pt) — PPE compliance disabled")
    ppe_stats = {
        "enabled": bool(ppe and ppe.available),
        "model": ppe.weight_path.name if ppe and ppe.available else None,
        "violation_frames": 0,
        "tracks_hit": Counter(),
    }

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline.fps = fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_events: list[dict] = []
    TROLLEY_DETS: list[tuple[float, tuple]] = []  # (t, bbox)
    TRACK_BBOXES: dict[int, tuple] = {}          # track_id -> most recent bbox
    blur_boxes: list[tuple] = []                 # face regions for the current frame
    det_counter = Counter()
    person_frames = 0
    object_frames = 0
    t0 = time.time()
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        ppe_violations = None
        blur_boxes = []

        # FSM/risk pipeline on non-person, non-carrier tracks
        pipeline_events = pipeline.process_frame(
            frame, timestamp=t, camera_id=slug,
            exclude_classes={"person", "trolley", "pallet", "wooden pallet", "cart", "hand truck", "dolly"},
        )

        for det in detector.last.detections:
            det_counter[det.class_name] += 1
            if det.class_name == "person":
                person_frames += 1
            elif det.class_name in {"pallet", "trolley", "cart", "hand truck", "dolly"}:
                if det.confidence >= 0.20:
                    TROLLEY_DETS.append((t, det.bbox))
            else:
                object_frames += 1
                if det.track_id is not None:
                    TRACK_BBOXES[det.track_id] = det.bbox

        evs = []
        for pe in pipeline_events:
            raw = {
                "behaviour_type": pe.behaviour_type,
                "timestamp": round(pe.timestamp, 3),
                "risk_score": round(pe.risk_score, 1),
                "tier": pe.tier,
                "confidence": round(pe.confidence, 3),
                "status": pe.status,
                "features": pe.features,
                "track_id": pe.track_id,
                "frame": n,
                "source": "fsm",
                "class": next((d.class_name for d in detector.last.detections
                               if d.track_id == pe.track_id), "?"),
            }
            raw_events.append(raw)
            evs.append(raw)

        cross_evs = cross.update(detector.last)
        for ce in cross_evs:
            raw_events.append(ce)
            evs.append(ce)

        # PPE compliance: second detector on the same frame, folded into the
        # same event schema -> risk scoring, confidence gate, reports, feed.
        # Runs on a stride (every PPE_STRIDE-th frame) to keep CPU throughput;
        # the 8s dedup window makes skipped frames harmless.
        # Face regions for blurring (from this frame's person boxes)
        if blur_enabled:
            blur_boxes = face_boxes_from_persons(
                [d for d in detector.last.detections if d.class_name == "person"]
            )

        if ppe is not None and ppe.available and n % PPE_STRIDE == 0:
            person_dets = [d for d in detector.last.detections if d.class_name == "person"]
            if person_dets:
                ppe_violations = ppe.check_persons(frame, person_dets)
                if ppe_violations:
                    ppe_stats["violation_frames"] += 1
                    ppe_stats["tracks_hit"].update(str(v.track_id) for v in ppe_violations)
                    ppe_events = ppe.build_events(ppe_violations, t)
                    raw_events.extend(ppe_events)
                    evs.extend(ppe_events)

        if evs and not args.no_clips:
            ann = annotate(frame, detector.last.detections, evs, ppe_violations=ppe_violations)
            # Responsible AI (plan §5): faces pixelated in stored clips by default.
            ann = maybe_blur_frame(ann, blur_boxes, blur_enabled)
            cv2.imwrite(str(clip_dir / f"event_frame_{n:06d}.jpg"), ann)

        n += 1
        if n % 500 == 0:
            print(f"    frame {n}/{total_frames} ({n/fps:.0f}s)  fps={n/(time.time()-t0):.1f}",
                  flush=True)

    cap.release()
    dur = time.time() - t0

    # ---- dedup: merge same behaviour+track within 8s into one event ----
    unique: list[dict] = []
    for e in sorted(raw_events, key=lambda x: x["timestamp"]):
        merged = False
        for u in unique:
            if (u["behaviour_type"] == e["behaviour_type"] and u["track_id"] == e["track_id"]
                    and abs(u["timestamp"] - e["timestamp"]) < 8.0):
                if e["risk_score"] > u["risk_score"]:
                    u["risk_score"] = e["risk_score"]
                    u["tier"] = e["tier"]
                if e["confidence"] > u["confidence"]:
                    u["confidence"] = e["confidence"]
                u["end_timestamp"] = max(u.get("end_timestamp", u["timestamp"]), e["timestamp"])
                merged = True
                break
        if not merged:
            unique.append(dict(e))
            unique[-1]["end_timestamp"] = e["timestamp"]

    # Drops only supersede thrown if the thrown candidate had low horizontal velocity (i.e. was a vertical fall)
    drop_times = [e["timestamp"] for e in unique if e["behaviour_type"] == EventType.PRODUCT_DROPPED.value]
    if drop_times:
        unique = [
            e for e in unique
            if not (e["behaviour_type"] == EventType.PRODUCT_THROWN.value
                    and any(abs(e["timestamp"] - dt) <= 2.0 for dt in drop_times)
                    and abs((e.get("features") or {}).get("throw_velocity", (0, 0))[0]) < 50.0)
        ]

    # ---- derive 8_dragging_no_trolley: heavy furniture/cupboard dragged without a trolley ----
    def _trolley_near(t, drag_bbox=None, window=2.5, max_dist=280.0):
        for (tt, tb) in TROLLEY_DETS:
            if abs(t - tt) <= window:
                if drag_bbox is None:
                    return tb[3] >= 480.0
                ix1 = max(drag_bbox[0], tb[0])
                iy1 = max(drag_bbox[1], tb[1])
                ix2 = min(drag_bbox[2], tb[2])
                iy2 = min(drag_bbox[3], tb[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                min_area = min((drag_bbox[2] - drag_bbox[0]) * (drag_bbox[3] - drag_bbox[1]),
                               (tb[2] - tb[0]) * (tb[3] - tb[1]))
                if min_area > 0 and (inter / min_area) > 0.40:
                    # Self-detection of the product's base/drawers/slats as trolley
                    continue
                dcx = (drag_bbox[0] + drag_bbox[2]) / 2.0
                dcy = (drag_bbox[1] + drag_bbox[3]) / 2.0
                tcx = (tb[0] + tb[2]) / 2.0
                tcy = (tb[1] + tb[3]) / 2.0
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist <= max_dist:
                    return True
        return False

    thrown_times = [e["timestamp"] for e in unique if e["behaviour_type"] == EventType.PRODUCT_THROWN.value]

    def _thrown_near(t, window=1.5):
        return any(abs(t - tt) <= window for tt in thrown_times)

    for e in list(unique):
        drag_bbox = (e.get("features") or {}).get("bbox") or TRACK_BBOXES.get(e.get("track_id"))
        if e["behaviour_type"] == EventType.PRODUCT_DRAGGED.value and not _trolley_near(e["timestamp"], drag_bbox):
            drag_dur = e.get("end_timestamp", e["timestamp"]) - e["timestamp"]
            bw = (drag_bbox[2] - drag_bbox[0]) if drag_bbox else 0
            bh = (drag_bbox[3] - drag_bbox[1]) if drag_bbox else 0
            area = bw * bh
            # 8_dragging_no_trolley applies when heavy assembled goods/cupboards/furniture are dragged without a trolley
            is_heavy = (e.get("class") in {"cupboard", "cabinet", "appliance", "furniture", "wardrobe"}
                        or area >= 80000 or (drag_dur >= 1.5 and bh >= 200))
            if is_heavy and drag_dur >= 0.8 and (drag_dur >= 1.5 or not _thrown_near(e["timestamp"])):
                derived = dict(e)
                derived["behaviour_type"] = EventType.DRAGGING_NO_TROLLEY.value
                derived["source"] = "derived_no_trolley"
                derived["confidence"] = round(e["confidence"] * 0.9, 3)
                if "end_timestamp" not in derived:
                    derived["end_timestamp"] = derived["timestamp"]
                unique.append(derived)
    unique = [e for e in unique if e.get("confidence", 0) >= 0.20]
    unique.sort(key=lambda x: x["timestamp"])

    # ---- clips for unique events (annotated: boxes + banner + blur) ----
    clips = []
    if not args.no_clips:
        for e in unique:
            start_t = max(0.0, e["timestamp"] - 1.5)
            end_t = e["timestamp"] + 1.5
            mp4 = clip_dir / f"{e['behaviour_type']}_t{e['timestamp']:.1f}s.mp4"
            ok = annotate_and_encode_clip(
                video_path, mp4, start_t, end_t, fps,
                detector, pipeline, ppe, blur_enabled, event=e,
            )
            if not ok:
                # No ffmpeg or encode failure — raw segment still preserves evidence.
                ok = extract_clip(video_path, mp4, start_t, end_t, fps)
            if ok:
                clips.append(str(mp4.relative_to(ROOT)))

    report = {
        "filename": name,
        "duration_s": round(total_frames / fps, 1),
        "frames": total_frames,
        "fps": round(fps, 2),
        "processing_fps": round(n / dur, 1),
        "detection_stats": {
            "person_frames": person_frames,
            "object_frames": object_frames,
            "by_class": dict(det_counter.most_common()),
        },
        "ground_truth": GROUND_TRUTH.get(name, []),
        "ppe": {
            "enabled": ppe_stats["enabled"],
            "model": ppe_stats["model"],
            "violation_frames": ppe_stats["violation_frames"],
            "tracks": dict(ppe_stats["tracks_hit"]),
        },
        "face_blur": {
            "enabled": blur_enabled,
            "policy": "faces pixelated in stored event frames by default; "
                      "raw mp4 extracts remain local-only source footage",
        },
        "raw_events": raw_events,
        "unique_events": unique,
        "clips": clips,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    detected = sorted({e["behaviour_type"] for e in unique})
    print(f"  {name[:55]:57s} {n} frames in {dur:.0f}s | "
          f"detected: {detected or 'NONE'} | gt: {report['ground_truth']}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=str, default=str(VIDEOS_DIR))
    ap.add_argument("--limit", type=int, default=0, help="process only first N videos")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--no-ppe", action="store_true", help="disable the PPE compliance detector")
    ap.add_argument("--no-blur", action="store_true", help="disable face blurring on stored frames")
    ap.add_argument("--ppe-model", type=str, default=None, help="path to PPE .pt weights (default: auto-find ml/ppe_best.pt)")
    args = ap.parse_args()

    v_path = Path(args.videos)
    if v_path.is_file():
        vids = [v_path]
    else:
        vids = sorted(v_path.glob("*.mp4"))
    if args.limit:
        vids = vids[:args.limit]
    print(f"Processing {len(vids)} videos with YOLO-World -> FSM -> risk -> gate", flush=True)

    all_reports = []
    for v in vids:
        if v.name not in SCENARIO_PROMPTS:
            # New/unknown video: process with the general warehouse prompt
            # (run_video resolves it) — dropping a file into argus/videos is
            # enough to onboard it.
            print(f"== {v.name}  (no scenario prompts — using general set)", flush=True)
        else:
            print(f"== {v.name}", flush=True)
        try:
            all_reports.append(run_video(v, args))
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}\n{traceback.format_exc()}")

    # Rebuild summary.json from ALL per-video reports on disk (not just the
    # ones processed in this invocation) so partial runs never leave a stale
    # summary behind for eval.py or seed_db.py to consume.
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    disk_reports = []
    for d in sorted(RESULTS_DIR.iterdir()):
        rp = d / "report.json" if d.is_dir() else None
        if rp and rp.exists():
            try:
                with open(rp, encoding="utf-8") as f:
                    disk_reports.append(json.load(f))
            except Exception:
                pass
    with open(summary_path, "w") as f:
        json.dump({"videos": disk_reports}, f, indent=2, default=str)
    print(f"\nWrote {len(all_reports)} reports -> {RESULTS_DIR}")
    print(f"Summary rebuilt from {len(disk_reports)} on-disk reports -> {summary_path}")
    print(f"Clips   -> {CLIPS_DIR}")


if __name__ == "__main__":
    main()