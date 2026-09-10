"""One-off repair: re-encode all existing event clips to browser-playable H.264.

The pipeline originally wrote clips with OpenCV's mp4v (FMP4) codec, which
Chrome/Safari cannot decode — the incident replay player stayed blank even
when the file was served. This script re-extracts each clip straight from the
source video with the same timestamps (no re-detection) and overwrites the
mp4v version in place.

Usage:  python ml/fix_clips_h264.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_on_videos import VIDEOS_DIR, extract_clip  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# run_on_videos.py sets ROOT = argus/ and CLIPS_DIR = ROOT/"clips"; report
# clip paths are relative to ROOT — so clips live in argus/clips/, not ml/.
ARGUS_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ARGUS_ROOT / "clips"


def main() -> None:
    fixed = missing_src = failed = 0
    for report_path in sorted(RESULTS_DIR.glob("*/report.json")):
        slug = report_path.parent.name
        report = json.loads(report_path.read_text(encoding="utf-8"))
        video = VIDEOS_DIR / report.get("filename", f"{slug}.mp4")
        if not video.exists():
            print(f"!! source video missing for {slug}: {video.name}")
            missing_src += 1
            continue
        for rel in report.get("clips", []):
            clip = ARGUS_ROOT / rel  # report stores paths relative to argus/
            if not clip.exists():
                continue
            # Clip filename carries the event timestamp: <behaviour>_t<sec>s.mp4
            try:
                t = float(clip.stem.split("_t")[1].rstrip("s"))
            except (IndexError, ValueError):
                continue
            out = clip.with_suffix(".h264.mp4")
            ok = extract_clip(video, out, max(0.0, t - 1.5), t + 1.5, report.get("fps", 30))
            if ok and out.exists() and out.stat().st_size > 10_000:
                clip.unlink(missing_ok=True)
                out.rename(clip)
                fixed += 1
            else:
                out.unlink(missing_ok=True)
                failed += 1
        print(f"{slug}: done")
    print(f"\nre-encoded: {fixed} | failed: {failed} | source missing: {missing_src}")


if __name__ == "__main__":
    main()
