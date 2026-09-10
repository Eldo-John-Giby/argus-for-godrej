"""Evaluation script — precision, recall, and false-positive rate on labeled clips.

Produces the real numbers for the "AI Performance" metrics.
A real number, even 78% precision on your own eval set, beats an unverified claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    """Evaluation metrics."""
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_predictions: int
    total_ground_truth: int


def evaluate_predictions(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> EvalResult:
    """Evaluate predictions against ground truth labels.

    Each prediction and ground truth entry should have:
    - behaviour_type: str
    - timestamp: float (seconds)
    - bbox: [x1, y1, x2, y2] (optional, for spatial matching)
    - label: str ("positive" | "negative")

    Matching criteria:
    - Same behaviour_type
    - Within iou_threshold spatial overlap (if bbox provided)
    - Within time tolerance (±0.5s)
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in predictions:
        matched = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue

            # Match on behaviour type
            if pred.get("behaviour_type") != gt.get("behaviour_type"):
                continue

            # Match on time (within 0.5s)
            time_diff = abs(pred.get("timestamp", 0) - gt.get("timestamp", 0))
            if time_diff > 0.5:
                continue

            # Match on spatial overlap (if bbox available)
            if "bbox" in pred and "bbox" in gt:
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou < iou_threshold:
                    continue

            matched_gt.add(i)
            matched = True
            break

        if matched:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return EvalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        true_positives=tp,
        false_positives=fp,
        true_negatives=0,  # not computed in this simplified version
        false_negatives=fn,
        total_predictions=len(predictions),
        total_ground_truth=len(ground_truth),
    )


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def load_labeled_data(labels_path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load labeled predictions and ground truth from a JSON file.

    Expected format:
    {
        "predictions": [...],
        "ground_truth": [...]
    }
    """
    path = Path(labels_path)
    if not path.exists():
        return [], []

    with open(path) as f:
        data = json.load(f)

    return data.get("predictions", []), data.get("ground_truth", [])


def evaluate_from_summary(summary_path: str | Path) -> dict:
    """Video-level evaluation from run_on_videos.py output.

    Ground truth is the behaviour list encoded in each judge video's name;
    predictions are the unique event types the pipeline emitted. Matching is
    per-video on behaviour type (video-level recall/precision), which is the
    metric that matters for the demo: did ARGUS flag the right violation
    in the right clip?

    Writes metrics back next to the summary as eval_metrics.json.
    """
    path = Path(summary_path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    tp = fp = fn = 0
    core_tp = core_fp = core_fn = 0
    ppe_event_count = 0
    per_video = []
    for report in data.get("videos", []):
        gt = set(report.get("ground_truth") or [])
        pred = {e["behaviour_type"] for e in report.get("unique_events") or []}
        # 13_ppe_noncompliance comes from the separate PPE detector and has no
        # ground-truth labels in the video names — score the core behaviours
        # without it so the PPE module isn't counted as false positives.
        core_pred = pred - {"13_ppe_noncompliance"}
        ppe_event_count += sum(
            1 for e in (report.get("unique_events") or [])
            if e["behaviour_type"] == "13_ppe_noncompliance"
        )
        v_tp = len(gt & pred)
        v_fp = len(pred - gt)
        v_fn = len(gt - pred)
        tp += v_tp
        fp += v_fp
        fn += v_fn
        core_tp += len(gt & core_pred)
        core_fp += len(core_pred - gt)
        core_fn += len(gt - core_pred)
        per_video.append({
            "video": report.get("filename"),
            "ground_truth": sorted(gt),
            "predicted": sorted(pred),
            "true_positives": v_tp,
            "false_positives": v_fp,
            "false_negatives": v_fn,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    core_precision = core_tp / (core_tp + core_fp) if (core_tp + core_fp) > 0 else 0.0
    core_recall = core_tp / (core_tp + core_fn) if (core_tp + core_fn) > 0 else 0.0
    core_f1 = 2 * core_precision * core_recall / (core_precision + core_recall) if (core_precision + core_recall) > 0 else 0.0
    metrics = {
        "level": "video",
        "videos_evaluated": len(per_video),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        # Core = the 12 ground-truth behaviours only (excludes the PPE module,
        # which has no ground-truth labels in the judge video names).
        "core_precision": round(core_precision, 3),
        "core_recall": round(core_recall, 3),
        "core_f1": round(core_f1, 3),
        "core_true_positives": core_tp,
        "core_false_positives": core_fp,
        "core_false_negatives": core_fn,
        "ppe_events": ppe_event_count,
        "per_video": per_video,
        "source": str(path),
    }
    out = path.parent / "eval_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_evaluation(labels_dir: str = "./clips") -> dict:
    """Run evaluation on all labeled clips in a directory."""
    labels_path = Path(labels_dir)
    all_predictions = []
    all_ground_truth = []

    for labels_file in labels_path.glob("*.json"):
        preds, gts = load_labeled_data(labels_file)
        all_predictions.extend(preds)
        all_ground_truth.extend(gts)

    if not all_predictions and not all_ground_truth:
        # Return demo metrics
        return {
            "precision": 0.78,
            "recall": 0.72,
            "f1": 0.75,
            "false_positive_rate": 0.22,
            "true_positives": 39,
            "false_positives": 11,
            "false_negatives": 15,
            "total_predictions": 50,
            "total_ground_truth": 54,
            "note": "Demo metrics — run with labeled data for real numbers",
        }

    result = evaluate_predictions(all_predictions, all_ground_truth)
    return {
        "precision": round(result.precision, 3),
        "recall": round(result.recall, 3),
        "f1": round(result.f1, 3),
        "false_positive_rate": round(result.false_positive_rate, 3),
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "total_predictions": result.total_predictions,
        "total_ground_truth": result.total_ground_truth,
    }


def _print_report(metrics: dict) -> None:
    header = [k for k in ("precision", "recall", "f1") if k in metrics]
    print(" ".join(f"{k}={metrics[k]:.3f}" for k in header) if header else "no metrics")
    for row in metrics.get("per_video", []):
        mark = "OK " if row["false_negatives"] == 0 and row["false_positives"] == 0 else "   "
        print(f"  {mark}{row['video'][:58]:60s} tp={row['true_positives']} "
              f"fp={row['false_positives']} fn={row['false_negatives']}")
        if row["false_negatives"]:
            missed = sorted(set(row["ground_truth"]) - set(row["predicted"]))
            print(f"       missed: {missed}")


if __name__ == "__main__":
    import sys

    summary = Path(__file__).resolve().parents[1] / "ml" / "results" / "summary.json"
    if len(sys.argv) > 1:
        summary = Path(sys.argv[1])
    metrics = evaluate_from_summary(summary)
    if metrics:
        _print_report(metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_video"}, indent=2))
    else:
        metrics = run_evaluation()
        print(json.dumps(metrics, indent=2))
