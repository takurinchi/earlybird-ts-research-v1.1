"""Create accuracy and post-t_s training-time comparisons for Ver.1.1."""
import os
from pathlib import Path

import pandas as pd

MPL_CONFIG = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_compaction_comparison(summary_csv, output_dir):
    data = pd.read_csv(summary_csv)
    required = {"ts", "accuracy", "training_strategy", "information_mode",
                "compact_train_seconds"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "status" in data:
        failed = int((data["status"] != "completed").sum())
        data = data[data["status"] == "completed"].copy()
    else:
        failed = 0
    if data.empty:
        (output_dir / "plot_status.json").write_text(
            '{"status":"not_generated","reason":"no completed comparison runs"}\n',
            encoding="utf-8")
        return
    data = data.sort_values("ts")

    plots = (
        ("accuracy", "Final test accuracy", "accuracy_by_ts.png"),
        ("compact_train_seconds", "Post-t_s training time (seconds)",
         "training_time_by_ts.png"),
    )
    for metric, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (strategy, mode), frame in data.groupby(["training_strategy", "information_mode"]):
            frame = frame.sort_values("ts")
            ax.plot(frame["ts"], frame[metric], marker="o",
                    label=f"{strategy} / {mode}")
        ax.set_xlabel("t_s")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    if failed:
        (output_dir / "plot_status.json").write_text(
            f'{{"status":"partial","failed_or_incomplete_runs":{failed}}}\n',
            encoding="utf-8")
