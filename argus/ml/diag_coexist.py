"""Dump sampled frames where 2+ objects coexist, with bboxes and track ids.

Usage: python ml/diag_coexist.py "<video name>" [sample_every_n_frames]
"""
import sys
from pathlib import Path
from collections import defaultdict

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_on_videos import SCENARIO_PROMPTS, WorldDetector, VIDEOS_DIR

name = sys.argv[1]
sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else 30

vid = VIDEOS_DIR / name
det = WorldDetector(SCENARIO_PROMPTS[name])

cap = cv2.VideoCapture(str(vid))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

coexist_frames = 0
n = 0
shown = 0
track_summary = defaultdict(lambda: [0, None, None])  # id -> [frames, first_cls, last_box]
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = n / fps
    dets = det.detect_and_track(frame, timestamp=t)
    objs = [d for d in dets.detections if d.class_name != "person" and d.track_id is not None]
    for d in objs:
        s = track_summary[d.track_id]
        s[0] += 1
        s[1] = s[1] or d.class_name
        s[2] = d.bbox
    if len(objs) >= 2:
        coexist_frames += 1
        if shown < 25:
            boxes = "  ".join(
                f"#{d.track_id}:{d.class_name[:9]}[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}]"
                for d in objs
            )
            print(f"t={t:6.1f}s n={len(objs)}  {boxes}")
            shown += 1
    n += 1
    if sample_every > 1:
        for _ in range(sample_every - 1):
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
    if not ok:
        break
cap.release()

print(f"\nframes with >=2 objects coexisting: {coexist_frames}")
print("long-lived object tracks (id: frames, class, last box):")
for tid, (cnt, cls, box) in sorted(track_summary.items(), key=lambda kv: -kv[1][0])[:12]:
    if cnt >= 5:
        print(f"  #{tid}: {cnt:4d} frames  {cls}  last={[round(v) for v in box]}")
