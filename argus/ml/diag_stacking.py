"""Find object-object stack pairs and how long they persist.

Usage: python ml/diag_stacking.py "<video name>" [max_frames]
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

# pair (idA,idB) -> frames where geometry holds; also dump per-frame object boxes
pair_frames = defaultdict(int)
pair_span = {}
obj_classes = defaultdict(set)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = n / fps
    dets = det.detect_and_track(frame, timestamp=t)
    objs = [d for d in dets.detections if d.class_name != "person" and d.track_id is not None]
    for i, a in enumerate(objs):
        obj_classes[a.track_id].add(a.class_name)
        for b in objs[i + 1:]:
            obj_classes[b.track_id].add(b.class_name)
            top, bot = (a, b) if a.bbox[1] < b.bbox[1] else (b, a)
            tx1, ty1, tx2, ty2 = top.bbox
            bx1, by1, bx2, by2 = bot.bbox
            band = by1 < ty2 < by1 + 0.5 * (by2 - by1)
            ov = min(tx2, bx2) - max(tx1, bx1)
            if band and ov > 0.3 * min(tx2 - tx1, bx2 - bx1):
                key = tuple(sorted((top.track_id, bot.track_id)))
                pair_frames[key] += 1
                pair_span.setdefault(key, [t, t])[1] = t
    n += 1
    if max_frames and n >= max_frames:
        break
cap.release()

print(f"processed {n} frames")
print(f"{'pair':>14} {'frames':>6} {'span_s':>7}  classes")
for key, cnt in sorted(pair_frames.items(), key=lambda kv: -kv[1])[:15]:
    a, b = key
    span = pair_span[key][1] - pair_span[key][0]
    ca = "/".join(sorted(obj_classes[a])) or "?"
    cb = "/".join(sorted(obj_classes[b])) or "?"
    print(f"{str(key):>14} {cnt:>6} {span:>7.1f}  #{a}:{ca}  |  #{b}:{cb}")
if not pair_frames:
    print("NO stack pairs found at all")
