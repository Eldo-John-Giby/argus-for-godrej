import sys
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_on_videos import SCENARIO_PROMPTS, WorldDetector, VIDEOS_DIR

def diag_video(vid_name):
    vid_path = VIDEOS_DIR / vid_name
    print(f"\n=================== {vid_name} ===================")
    det = WorldDetector(SCENARIO_PROMPTS[vid_name])
    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total}, fps: {fps}")

    track_history = defaultdict(list)
    person_bottoms = defaultdict(list)
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        dets = det.detect_and_track(frame, timestamp=t)
        persons = [d for d in dets.detections if d.class_name == "person"]
        for p in persons:
            person_bottoms[n].append(p.bbox[3])

        for d in dets.detections:
            if d.track_id is not None:
                track_history[d.track_id].append((n, t, d.class_name, d.bbox, d.confidence))
        n += 1

    cap.release()

    # Analyze tracks
    for tid, hist in sorted(track_history.items(), key=lambda x: -len(x[1])):
        if len(hist) < 5:
            continue
        classes = Counter([h[2] for h in hist])
        main_cls = classes.most_common(1)[0][0]
        start_f, end_f = hist[0][0], hist[-1][0]
        start_t, end_t = hist[0][1], hist[-1][1]
        
        # Calculate velocities
        vels = []
        for i in range(1, len(hist)):
            dt = hist[i][1] - hist[i-1][1]
            if dt > 0 and (hist[i][0] - hist[i-1][0] == 1):
                c1 = ((hist[i-1][3][0] + hist[i-1][3][2])/2, (hist[i-1][3][1] + hist[i-1][3][3])/2)
                c2 = ((hist[i][3][0] + hist[i][3][2])/2, (hist[i][3][1] + hist[i][3][3])/2)
                vx = (c2[0] - c1[0]) / dt
                vy = (c2[1] - c1[1]) / dt
                vels.append((vx, vy, (vx**2 + vy**2)**0.5, hist[i][3][3]))
        
        max_speed = max([v[2] for v in vels]) if vels else 0.0
        max_vy = max([v[1] for v in vels]) if vels else 0.0
        min_vy = min([v[1] for v in vels]) if vels else 0.0
        max_vx = max([abs(v[0]) for v in vels]) if vels else 0.0
        avg_bottom = np.mean([h[3][3] for h in hist])
        
        print(f"Track #{tid:3d} ({main_cls:15s}): frames={len(hist):3d} [{start_f:4d}-{end_f:4d}] "
              f"t=[{start_t:4.1f}s-{end_t:4.1f}s] max_spd={max_speed:5.1f} max_vx={max_vx:5.1f} "
              f"max_vy={max_vy:5.1f} min_vy={min_vy:5.1f} avg_bot={avg_bottom:5.1f}")

if __name__ == '__main__':
    for name in sys.argv[1:]:
        diag_video(name)
