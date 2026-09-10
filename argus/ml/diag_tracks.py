"""Dump per-track stats for one video: lifetimes, speeds, bottom_y.

Usage: python ml/diag_tracks.py "<video name>" [max_frames]
"""
import sys
from pathlib import Path
from collections import defaultdict

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_on_videos import SCENARIO_PROMPTS, WorldDetector, VIDEOS_DIR

name = sys.argv[1]
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 0

vid = VIDEOS_DIR / name
det = WorldDetector(SCENARIO_PROMPTS[name])

cap = cv2.VideoCapture(str(vid))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

tracks = defaultdict(list)  # track_id -> list of (t, cx, cy, bottom_y, conf, cls)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = n / fps
    dets = det.detect_and_track(frame, timestamp=t)
    for d in dets.detections:
        if d.track_id is None:
            continue
        x1, y1, x2, y2 = d.bbox
        tracks[d.track_id].append((t, (x1 + x2) / 2, (y1 + y2) / 2, y2, d.confidence, d.class_name))
    n += 1
    if max_frames and n >= max_frames:
        break
cap.release()

print(f"processed {n} frames, {len(tracks)} track ids")
print(f"{'id':>5} {'cls':<14} {'frames':>6} {'dur_s':>6} {'conf':>5} {'btmY~':>6} {'spd_max':>8} {'spd_mean':>8}")
for tid, rows in sorted(tracks.items(), key=lambda kv: -len(kv[1]))[:25]:
    if len(rows) < 4:
        continue
    cls = rows[0][5]
    dur = rows[-1][0] - rows[0][0]
    conf = sum(r[4] for r in rows) / len(rows)
    btm = sum(r[3] for r in rows) / len(rows)
    speeds = []
    for a, b in zip(rows, rows[1:]):
        dt = b[0] - a[0]
        if dt > 0:
            sp = ((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 / dt
            speeds.append(sp)
    smax = max(speeds) if speeds else 0
    smean = sum(speeds) / len(speeds) if speeds else 0
    print(f"{tid:>5} {cls:<14} {len(rows):>6} {dur:>6.1f} {conf:>5.2f} {btm:>6.0f} {smax:>8.0f} {smean:>8.0f}")
