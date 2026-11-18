# Ver.1.1 train-time compaction comparison

Dense detector: `20261118T144148370595_baseline_seed1` (t_m=21). Paired Dense-late-compaction and train-time-compaction runs are listed in `summary.csv`; epoch metrics are in `metrics_all_epochs.csv`. The dense baseline's per-epoch candidate channel masks are in `baseline/masks/epoch_mask_matrix.csv` as one binary vector per layer and completed epoch. Failed runs are retained with their status and errors, and plots include completed runs only. Large checkpoints are kept in the project run folders and are not duplicated here.


Version: 1.12. Target retained BN channels: 80% (pruning ratio 0.2).
Training and measurement settings inherit Ver.1.1. See source_snapshot and provenance.json.
