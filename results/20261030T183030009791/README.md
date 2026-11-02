# Ver.1.1 train-time compaction comparison

Early-Bird baseline: `20261026T130454_baseline_seed1` (`t_m=28`).

## Record order

The run-level `summary.csv`, `metrics_all_epochs.csv`, and `RUN_INDEX.csv` use the same paired order: `online` then `posthoc`; within each mode C1 through C4; within each condition Dense control first, then train-time physical compaction. `compaction_comparison.csv` keeps each pair on one row with Dense and compact fields side by side.

`summary.csv` contains one finalized row per run. `metrics_all_epochs.csv` contains the 160 epoch records for each run in the same order. `RUN_INDEX.csv` maps each sequence number to its run folder and final metrics. `manifest.json` lists run IDs in that order.

All 16 runs completed. The large model weight checkpoints are not included; per-run logs, configurations, epoch metrics, summaries, environment records, and transition masks are included.
