"""Collect per-run metrics while preserving missing values and failed runs."""
from pathlib import Path
import pandas as pd


def aggregate(root, output=None):
    files = list(Path(root).glob("*/metrics.csv"))
    frames = [pd.read_csv(p).assign(source_experiment=p.parent.name) for p in files if p.stat().st_size]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result
