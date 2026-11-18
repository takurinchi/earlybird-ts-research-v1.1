"""Join run summaries and compute metric deltas relative to a baseline."""
import json
from pathlib import Path
import pandas as pd


def compare(root, baseline_id, output=None):
    rows = []
    for p in Path(root).glob("*/summary.json"):
        summary = json.loads(p.read_text(encoding="utf-8"))
        summary.setdefault("experiment_id", p.parent.name)
        rows.append(summary)
    frame = pd.DataFrame(rows)
    if frame.empty or baseline_id not in set(frame.get("experiment_id", [])):
        raise ValueError(f"Baseline {baseline_id!r} summary not found")
    baseline = frame.loc[frame.experiment_id == baseline_id].iloc[0]
    for metric in ("parameter_count", "flops", "model_size_bytes"):
        if metric in frame and pd.notna(baseline.get(metric)) and baseline[metric] > 0:
            frame[metric + "_reduction_percent"] = (baseline[metric] - frame[metric]) / baseline[metric] * 100
    if output:
        frame.to_csv(output, index=False)
    return frame
