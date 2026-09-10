"""PPE compliance detection — second YOLO detector over the same frames.

Drop-in module: point ``ultralytics.YOLO`` at a pretrained helmet/vest weight
(e.g. a Workspace-Safety YOLOv8 PPE model), run it on the frames the main
YOLO-World pass already processes, and fold any NO-* boxes overlapping a
tracked person into the existing behaviour/risk pipeline as
``13_ppe_noncompliance``.

The module is optional end-to-end:
- weight file missing  -> available=False, runner prints one notice, zero cost
- model load failure   -> same
- detection failure    -> returns [] for that frame, pipeline keeps running

Class mapping is tolerant: any detection class containing "no" AND one of
helmet/hardhat/vest/mask counts as a violation box ("NO-Hardhat",
"NO-Safety Vest", "NO-Hardhat", "no helmet"...). Positive classes
("Hardhat", "Safety Vest") are used as mitigating evidence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running this file directly: python ppe_detector.py <frame.jpg>
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.risk.confidence_gate import ConfidenceInput, apply_confidence_gate
from app.risk.scoring import RiskInput, score_event
from app.behaviour.taxonomy import load_taxonomy

# New behaviour key — appended to taxonomy.yaml alongside the 12 existing ones.
PPE_BEHAVIOUR = "13_ppe_noncompliance"

VIOLATION_KEYWORDS = ("hardhat", "helmet", "vest", "mask")
MITIGATING_KEYWORDS = ("hardhat", "helmet", "vest")

DEFAULT_WEIGHT_CANDIDATES = (
    "ppe_best.pt",          # canonical name expected in argus/ml/ (or repo root)
    "ppe_model.pt",
    "best_ppe.pt",
    "best.pt",              # generic release-asset name (Workspace-Safety repo)
)


@dataclass
class PPEViolation:
    """One person-frame PPE non-compliance observation."""
    track_id: Optional[int]
    missing: list[str]           # e.g. ["hardhat", "vest"]
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class PPEDetections:
    """Raw PPE detections for a single frame (pre person-association)."""
    violations: list[tuple[str, tuple[float, float, float, float], float]] = field(default_factory=list)
    positive: list[tuple[str, tuple[float, float, float, float], float]] = field(default_factory=list)


class PPEDetector:
    """Optional second YOLO detector for helmet/vest compliance.

    Usage (already wired in run_on_videos.run_video):
        ppe = PPEDetector()                # finds weight automatically
        if ppe.available: ...
        violations = ppe.check_persons(frame, person_dets)
        events = ppe.build_events(violations, timestamp, taxonomy)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device

        self.weight_path: Optional[Path] = None
        self.available = False
        self._model = None

        if model_path is None:
            model_path = self._find_weight()
        if model_path:
            self._try_load(Path(model_path))

    # ------------------------------------------------------------- loading

    @staticmethod
    def _find_weight() -> Optional[str]:
        """Look for a PPE weight in argus/ml/, repo root, and env override."""
        import os

        env_path = os.getenv("PPE_MODEL_PATH")
        here = Path(__file__).resolve().parent
        root = here.parent
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        for name in DEFAULT_WEIGHT_CANDIDATES:
            candidates += [here / name, root / name]
        for c in candidates:
            if c.is_file():
                return str(c)
        return None

    def _try_load(self, path: Path) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(path))
            self.weight_path = path
            self.available = True
        except Exception as exc:  # ImportError, bad weights, etc.
            print(f"[ppe] PPE detector unavailable ({path.name}): {exc}")

    # ---------------------------------------------------------- inference

    def detect_frame(self, frame: np.ndarray) -> PPEDetections:
        """Run the PPE model on one frame; split boxes into violations vs positives."""
        out = PPEDetections()
        if not self.available or self._model is None:
            return out
        try:
            results = self._model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception:
            return out

        if not results or results[0].boxes is None:
            return out

        names = results[0].names
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = str(names.get(cls_id, f"class_{cls_id}")).lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            bbox = (x1, y1, x2, y2)

            is_neg = ("no" in cls_name or "without" in cls_name) and any(
                k in cls_name for k in VIOLATION_KEYWORDS
            )
            if is_neg:
                out.violations.append((cls_name, bbox, conf))
            elif any(k in cls_name for k in MITIGATING_KEYWORDS):
                out.positive.append((cls_name, bbox, conf))
        return out

    # --------------------------------------------------- person association

    def check_persons(
        self,
        frame: np.ndarray,
        person_dets,  # list[Detection] from the main detector
        min_head_overlap: float = 0.30,
        min_torso_overlap: float = 0.20,
    ) -> list[PPEViolation]:
        """Associate NO-* boxes with tracked persons.

        A missing hardhat/helmet box must overlap the person's head region
        (top 30% of the person bbox); a missing vest overlaps the torso
        (30–75% vertical band). Returns one violation per person per frame,
        aggregating everything missing on that person.
        """
        out = self.detect_frame(frame)
        violations: list[PPEViolation] = []

        for p in person_dets:
            if p.class_name != "person" or p.confidence < 0.35:
                continue
            px1, py1, px2, py2 = p.bbox
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)

            head_band = (py1, py1 + 0.30 * ph)
            torso_band = (py1 + 0.30 * ph, py1 + 0.75 * ph)

            missing: list[str] = []
            worst_conf = 0.0

            for cls_name, bbox, conf in out.violations:
                bx1, by1, bx2, by2 = bbox
                overlap_x = min(px2, bx2) - max(px1, bx1)
                overlap_y = min(py2, by2) - max(py1, by1)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                is_head_gear = ("hardhat" in cls_name or "helmet" in cls_name) and "no" in cls_name
                band = head_band if is_head_gear else torso_band
                threshold = min_head_overlap if is_head_gear else min_torso_overlap

                # Two box granularities exist across PPE models:
                #  - head-sized NO-boxes (e.g. head_nohelmet): assign when the box
                #    center sits inside the person's head/torso band
                #  - full-region NO-boxes (e.g. NO-Safety Vest): assign by
                #    band-coverage fraction >= threshold
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                center_in_band = (
                    (band[0] <= bcy <= band[1])
                    and (px1 - 0.1 * pw <= bcx <= px2 + 0.1 * pw)
                )

                band_h = max(1.0, band[1] - band[0])
                oy_start = max(py1, by1)
                oy_end = min(py2, by2)
                covered = min(oy_end, band[1]) - max(oy_start, band[0])
                frac = (overlap_x / pw) * (max(0.0, covered) / band_h)

                if center_in_band or frac >= threshold:
                    if is_head_gear:
                        missing.append("hardhat")
                    elif "vest" in cls_name:
                        missing.append("vest")
                    elif "mask" in cls_name:
                        missing.append("mask")
                    worst_conf = max(worst_conf, conf)

            # Mitigating evidence: a confident positive helmet/vest on the
            # same region cancels a flaky NO- box from the opposite class set.
            if missing:
                for cls_name, bbox, conf in out.positive:
                    bx1, by1, bx2, by2 = bbox
                    overlap_x = min(px2, bx2) - max(px1, bx1)
                    overlap_y = min(py2, by2) - max(py1, by1)
                    if overlap_x <= 0 or overlap_y <= 0:
                        continue
                    if "vest" in cls_name and "vest" in missing and conf > 0.5:
                        # positive vest detected by the same model — trust it over NO-
                        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                        if px1 <= cx <= px2 and torso_band[0] <= cy <= torso_band[1]:
                            missing.remove("vest")
                    if ("helmet" in cls_name or "hardhat" in cls_name) and "hardhat" in missing and conf > 0.5:
                        cy = (by1 + by2) / 2
                        if head_band[0] <= cy <= head_band[1]:
                            missing.remove("hardhat")

            if missing:
                violations.append(PPEViolation(
                    track_id=p.track_id,
                    missing=missing,
                    bbox=p.bbox,
                    confidence=max(worst_conf, p.confidence * 0.8),
                ))
        return violations

    # -------------------------------------------------------- event build

    def build_events(self, violations: list[PPEViolation], timestamp: float) -> list[dict]:
        """Convert violations into pipeline-style event dicts (risk + gate applied)."""
        taxonomy = load_taxonomy()
        severity = taxonomy.get(PPE_BEHAVIOUR, {}).get("severity_base", 60)
        events = []
        for v in violations:
            risk = score_event(RiskInput(
                behaviour_severity=severity,
                velocity_at_contact=0.0,
                mass_proxy=1.0,
                fragility_multiplier=1.0,
                stack_tilt_angle=0.0,
                top_bottom_footprint_ratio=1.0,
                recurrence_count=1,
                composite_confidence=v.confidence,
            ))
            gate = apply_confidence_gate(ConfidenceInput(
                detector_confidence=v.confidence,
                track_stability=0.8,  # person tracks are already stable via ByteTrack
                vlm_agreement=-1,
            ))
            events.append({
                "behaviour_type": PPE_BEHAVIOUR,
                "timestamp": timestamp,
                "risk_score": round(risk.score, 1),
                "tier": risk.tier,
                "confidence": round(gate.composite_confidence, 3),
                "status": gate.status,
                "features": {
                    "missing_ppe": v.missing,
                    "source_model": self.weight_path.name if self.weight_path else "unknown",
                    "domain_note": "trained on construction-site imagery",
                },
                "track_id": v.track_id,
                "source": "ppe",
            })
        return events


def _sanity_check_cli() -> None:
    """python ppe_detector.py <image> — validate PPE weights on a real frame
    before trusting them for the demo (construction-domain caveat check)."""
    import sys

    import cv2

    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python ppe_detector.py <image_or_video> [frame_index]")
        sys.exit(1)
    source = sys.argv[1]
    if source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"could not read frame from {source}")
    else:
        frame = cv2.imread(source)
        if frame is None:
            sys.exit(f"could not read image {source}")

    ppe = PPEDetector()
    if not ppe.available:
        sys.exit("PPE weight not found — put the model at ml/ppe_best.pt (or set PPE_MODEL_PATH)")
    print(f"model: {ppe.weight_path}")
    dets = ppe.detect_frame(frame)
    print(f"violations: {[(c, [round(x) for x in b], round(f, 2)) for c, b, f in dets.violations]}")
    print(f"positives:  {[(c, [round(x) for x in b], round(f, 2)) for c, b, f in dets.positive]}")
    for c, b, _ in dets.violations + dets.positive:
        x1, y1, x2, y2 = [int(v) for v in b]
        color = (0, 0, 255) if "no" in c else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, c, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    out = "ppe_sanity_check.jpg"
    cv2.imwrite(out, frame)
    print(f"annotated -> {out}")


if __name__ == "__main__":
    _sanity_check_cli()
