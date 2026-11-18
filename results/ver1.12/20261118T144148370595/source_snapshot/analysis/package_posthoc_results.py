"""Add a completed posthoc sweep to the curated, Git-tracked result bundle."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.plot_ts_comparison import plot_ts_comparison


BUNDLE = ROOT / "outputs" / "results" / "online_c4_20261026"
CONDITIONS = (
    ("C1_tm_minus_10_posthoc", "C1_before"),
    ("C2_tm_posthoc", "C2_near"),
    ("C3_tm_plus_10_posthoc", "C3_after"),
    ("C4_fixed_ts_40_posthoc", "C4_fixed"),
)


def package(run_id):
    summary_rows = pd.read_csv(BUNDLE / "summary.csv").to_dict(orient="records")
    epoch_frames = []
    for label, source_label in (
        ("baseline", "baseline"),
        ("C1_tm_minus_10", "C1_tm_minus_10"),
        ("C2_tm", "C2_tm"),
        ("C3_tm_plus_10", "C3_tm_plus_10"),
        ("C4_fixed_ts_40", "C4_fixed_ts_40"),
    ):
        metrics = pd.read_csv(BUNDLE / "runs" / label / "metrics.csv")
        metrics["source_experiment"] = metrics["experiment_id"]
        metrics["condition_label"] = source_label
        epoch_frames.append(metrics)

    run_map = {}
    for label, source_label in CONDITIONS:
        experiment_id = f"{run_id}_{source_label}_posthoc_seed1"
        source = ROOT / "outputs" / experiment_id
        source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(source / "metrics.csv")
        if source_summary.get("status") != "completed" or len(metrics) != 160:
            raise ValueError(f"Incomplete posthoc run: {experiment_id}")
        if source_summary.get("information_mode") != "posthoc":
            raise ValueError(f"Unexpected information mode in {experiment_id}")

        destination = BUNDLE / "runs" / label
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", "environment.json", "run.log", "metrics.csv"):
            shutil.copy2(source / name, destination / name)
        shutil.copy2(source / "masks" / "transition_mask.pt", destination / "transition_mask.pt")

        summary = dict(source_summary)
        summary["condition"] = label
        summary["checkpoint"] = None
        summary["checkpoint_omitted_from_git"] = True
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary_rows.append(summary)

        metrics["source_experiment"] = experiment_id
        metrics["condition_label"] = label
        epoch_frames.append(metrics)
        run_map[label] = experiment_id

    # The dense baseline is also the shared source run for the posthoc mask.
    # Keep an explicitly labeled summary row so readers can identify its role.
    summary_rows = [
        row for row in summary_rows
        if row.get("condition") != "posthoc_dense_detector_shared"
    ]
    dense_source = next(row for row in summary_rows if row.get("condition") == "baseline")
    posthoc_dense_summary = dict(dense_source)
    posthoc_dense_summary["condition"] = "posthoc_dense_detector_shared"
    summary_rows.append(posthoc_dense_summary)

    pd.DataFrame(summary_rows).to_csv(BUNDLE / "summary.csv", index=False)
    pd.concat(epoch_frames, ignore_index=True).to_csv(
        BUNDLE / "metrics_all_epochs.csv", index=False)
    comparison_csv = ROOT / "outputs" / f"ts_comparison_{run_id}.csv"
    shutil.copy2(comparison_csv, BUNDLE / "posthoc_comparison.csv")
    plot_ts_comparison(BUNDLE / "summary.csv", BUNDLE / "accuracy_by_ts.png", metric="accuracy")

    manifest_path = BUNDLE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["posthoc_run_ids"] = run_map
    manifest["posthoc_comparison_csv"] = "posthoc_comparison.csv"
    manifest["included_conditions"] = [
        "baseline", "posthoc_dense_detector_shared",
        "C1_tm_minus_10", "C2_tm", "C3_tm_plus_10", "C4_fixed_ts_40",
        "C1_tm_minus_10_posthoc", "C2_tm_posthoc", "C3_tm_plus_10_posthoc", "C4_fixed_ts_40_posthoc",
    ]
    manifest["information_modes"] = ["online", "posthoc"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (BUNDLE / "README.md").write_text(
        "# Online and posthoc C1-C4 result bundle\n\n"
        "CIFAR-10 / VGG16-BN, seed 1. The dense baseline detected t_m=28. "
        "Online and posthoc each compare C1=t_m-10 (18), C2=t_m (28), "
        "C3=t_m+10 (38), and C4=fixed t_s=40.\n\n"
        "The bundle contains configuration, environment metadata, logs, per-epoch "
        "metrics, summaries, mask traces, selected masks, aggregate CSVs, and an "
        "accuracy plot comparing information modes. Each per-run metrics.csv also "
        "contains final test and compact-model metrics in final_* columns. Large "
        "model checkpoints are excluded from Git.\n\n"
        "The summary includes `posthoc_dense_detector_shared`, an explicitly labeled "
        "reference to the shared dense baseline run that detected t_m=28 and supplied "
        "the fixed posthoc mask; it is the same experiment as `baseline`, not an "
        "additional training run.\n\n"
        "All four online and all four posthoc conditions completed 160 epochs and "
        "final test evaluation.\n",
        encoding="utf-8")
    print(f"Packaged {len(run_map)} posthoc runs into {BUNDLE}")
    print(f"Total summary rows: {len(summary_rows)}; epoch rows: {sum(len(x) for x in epoch_frames)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    package(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
