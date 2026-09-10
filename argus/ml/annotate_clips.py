"""Burn YOLO-style annotations into existing event clips.

The pipeline's stored clips were raw source segments; the detections existed
only as separate keyframe JPGs. This script re-runs the detector inside each
event's ±1.5s window and re-encodes the clip with:

  - object/person boxes + track ids (cyan / amber)
  - red NO-HARDHAT style boxes from the PPE compliance module
  - face pixelation (same policy as the pipeline, plan §5)
  - a banner:  behaviour_type  risk=NN.N

Timestamps and clip filenames are unchanged, so reports/DB stay valid.
Run after any full pipeline pass:

    python ml/annotate_clips.py
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_on_videos import (  # noqa: E402
    CLIPS_DIR,
    PPE_STRIDE,
    SCENARIO_PROMPTS,
    VIDEOS_DIR,
    WorldDetector,
    _ffmpeg_exe,
    face_boxes_from_persons,
    maybe_blur_frame,
)
from ppe_detector import PPEDetector  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# BGR colors matching the pipeline's annotate()
PERSON_COLOR = (255, 128, 0)
OBJECT_COLOR = (0, 200, 255)
PPE_COLOR = (0, 0, 255)


def draw_dets(img, dets, ppe_violations, banner=None):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        color = PERSON_COLOR if d.class_name == "person" else OBJECT_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.class_name} #{d.track_id}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for v in ppe_violations or []:
        x1, y1, x2, y2 = [int(c) for c in v.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), PPE_COLOR, 2)
        label = "NO-" + "/NO-".join(m.upper() for m in v.missing) + f" #{v.track_id}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, PPE_COLOR, 2)
    if banner:
        cv2.putText(img, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PPE_COLOR, 2)
    return img


def reannotate_clip(video_path, out_mp4, start_t, end_t, fps, detector, pipeline, ppe,
                    blur_enabled, banner):
    """Detect + annotate + encode one event window as H.264."""
    cap = cv2.VideoCapture(str(video_path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        print("  !! no ffmpeg — cannot re-encode")
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
    while written < int((end_t - start_t) * vfps):
        ok, frame = cap.read()
        if not ok:
            break
        t = start_t + written / vfps

        pipeline.process_frame(
            frame, timestamp=t, camera_id="reannotate",
            exclude_classes={"person", "trolley", "pallet", "wooden pallet",
                             "cart", "hand truck", "dolly"},
        )
        dets = detector.last.detections
        persons = [d for d in dets if d.class_name == "person"]

        violations = []
        if ppe is not None and ppe.available and persons:
            violations = ppe.check_persons(frame, persons)

        img = draw_dets(frame.copy(), dets, violations, banner=banner)
        if blur_enabled:
            img = maybe_blur_frame(img, face_boxes_from_persons(persons), blur_enabled)

        proc.stdin.write(img.tobytes())
        written += 1

    cap.release()
    proc.stdin.close()
    ok = proc.wait(timeout=120) == 0 and written > 5
    if not ok:
        out_mp4.unlink(missing_ok=True)
    return ok


def is_h264(path: Path) -> bool:
    """True if the file already carries an H.264 stream (already annotated)."""
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return "Video: h264" in (proc.stderr or "")
    except Exception:
        return False


def main() -> None:
    # Usage: python annotate_clips.py [slug-substring] [behaviour-substrings,comma] [--force]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    only = args[0] if args else ""
    behFilters = [b for b in (args[1].split(",") if len(args) > 1 else []) if b]
    ppe = PPEDetector()
    blur_enabled = True
    total = done = skipped = failed = 0

    for report_path in sorted(RESULTS_DIR.glob("*/report.json")):
        slug = report_path.parent.name
        if only and only.lower() not in slug.lower():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        video = VIDEOS_DIR / report.get("filename", f"{slug}.mp4")
        if not video.exists():
            print(f"!! source missing for {slug}")
            continue

        detector = WorldDetector(SCENARIO_PROMPTS.get(slug, SCENARIO_PROMPTS.get(
            report.get("filename", ""), "person, box, carton, trolley, pallet")))
        from app.pipeline import ProcessingPipeline  # noqa: E402  (path bootstrap lives in runner import)
        pipeline = ProcessingPipeline(detector=detector)
        fps = report.get("fps", 30)

        print(f"{slug}:")
        for ev in report.get("unique_events", []):
            if behFilters and not any(b in ev["behaviour_type"] for b in behFilters):
                continue
            total += 1
            t = float(ev["timestamp"])
            clip = CLIPS_DIR / slug / f"{ev['behaviour_type']}_t{t:.1f}s.mp4"
            if not clip.exists():
                skipped += 1
                continue
            if is_h264(clip) and not force:
                skipped += 1
                continue
            banner = f"{ev['behaviour_type']}  risk={ev['risk_score']}"
            out = clip.with_suffix(".ann.mp4")
            ok = reannotate_clip(
                video, out, max(0.0, t - 1.5), t + 1.5, fps,
                detector, pipeline, ppe, blur_enabled, banner,
            )
            if ok and out.stat().st_size > 10_000:
                clip.unlink(missing_ok=True)
                out.rename(clip)
                done += 1
            else:
                out.unlink(missing_ok=True)
                failed += 1
        print(f"  done ({done} annotated so far)")

    print(f"\nannotated: {done} | skipped (no file): {skipped} | failed: {failed} / {total}")


if __name__ == "__main__":
    main()
