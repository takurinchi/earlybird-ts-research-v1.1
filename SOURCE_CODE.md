# Source code and file inventory

この文書は`source_snapshot/`に含まれる実験コード・設定・計画書の役割と入口を示します。スナップショットは実験run `20261030T183030009791`に対応するVer.1.1作業環境から作成しました。

## 実行入口と設定の継承

- 本実験の入口: `source_snapshot/run_experiments.py`
- 設定継承: `configs/ts_compaction.yaml` → `configs/earlybird.yaml` → `configs/default.yaml`
- `main.py`は設定検査と空runディレクトリ準備の補助コードで、Ver.1.1の本実験入口ではありません。
- 再現コマンドと条件概要は[README](README.md)を参照してください。

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `.gitignore` | 結果bundleを追跡し、仮想環境・CIFAR-10データ・生成物を除外する規則。 |
| `analysis/aggregate_results.py` | run単位CSVの集約。 |
| `analysis/compare_experiments.py` | 複数runのsummary比較。 |
| `analysis/package_posthoc_results.py` | posthoc実験結果bundle作成。 |
| `analysis/plot_compaction_comparison.py` | Ver.1.1精度・時間plot生成。 |
| `analysis/plot_mask_distance.py` | Early-Birdマスク距離plot生成。 |
| `analysis/plot_ts_comparison.py` | t_s比較plot生成。 |
| `cifar_compaction_experiment.py` | Ver.1.1のペア実験、物理縮小、時間・精度集計、bundle作成。 |
| `cifar_experiment.py` | 共通のモデル初期化、baseline/mask取得、評価、checkpoint・epoch記録。 |
| `configs/baseline.yaml` | Dense baseline用のVer.1設定。 |
| `configs/default.yaml` | 基本条件。モデル、160 epoch、batch size、SGD、学習率schedule、選択率。 |
| `configs/earlybird.yaml` | Early-Bird検出条件。default.yamlを継承し、epsilonとmask windowを定義。 |
| `configs/ts_compaction.yaml` | Ver.1.1比較軸、C1〜C4、測定頻度、評価、checkpoint方針。 |
| `configs/ts_posthoc.yaml` | Ver.1のposthoc t_s比較条件。 |
| `configs/ts_sweep.yaml` | Ver.1のonline/posthoc t_s sweep設定。 |
| `data/__init__.py` | データセット読み込み。CIFAR-10の本体データは含まない。 |
| `data/cifar10.py` | CIFAR-10取得、augmentation、train/validation/test DataLoader生成。 |
| `data/datasets.py` | 設定指定のdataset loader factory解決。 |
| `docs/research_plan_ver1.1_train_time_compaction.md` | 研究目的、比較設計、実験条件、記録要件。 |
| `earlybird/__init__.py` | Early-Bird package marker。 |
| `earlybird/mask_distance.py` | マスク間距離の計算。 |
| `earlybird/stability_detector.py` | 複数epochのマスク安定性とt_m判定。 |
| `evaluation/__init__.py` | evaluation package marker。 |
| `evaluation/classification.py` | loss、accuracy、macro-F1の評価。 |
| `evaluation/flops.py` | モデルFLOPs計測。 |
| `evaluation/inference_time.py` | 推論レイテンシ測定。 |
| `evaluation/model_size.py` | checkpointのbyteサイズ計測。 |
| `evaluation/parameter_count.py` | パラメータ数とゼロ重み率。 |
| `evaluation/structured_metrics.py` | 縮小後VGGのパラメータ数・FLOPs推定。 |
| `evaluation/training_time.py` | 学習時間計測補助。 |
| `experiment_plan.json` | 旧Ver.1系計画のJSON。Ver.1.1本実験の正本ではない。 |
| `experiment_plan_v1_1.json` | Ver.1.1実行前に出力した計画確認JSON。実測結果はbundle内manifestを参照。 |
| `main.py` | 設定の妥当性確認とrunディレクトリ準備用の旧補助入口。Ver.1.1本実験にはrun_experiments.pyを使う。 |
| `models/__init__.py` | models package marker。 |
| `models/cifar_vgg.py` | CIFAR VGG16-BN定義とチャネルindexによる物理縮小。 |
| `models/model_factory.py` | モデルfactory解決。 |
| `pruning/__init__.py` | pruning package marker。 |
| `pruning/channel_manager.py` | Dense対照のBN出力ゲート。 |
| `pruning/mask_generator.py` | BN gamma magnitudeからチャネルmask生成。 |
| `pruning/mask_manager.py` | 一般的なmask適用補助。 |
| `pruning/mask_utils.py` | mask保存・読込。 |
| `pruning/ts_controller.py` | t_s移行制御補助。 |
| `requirements.txt` | Python依存パッケージ。CUDA版PyTorchはsetup_windows.ps1で別途指定。 |
| `run_experiments.py` | 実験計画の表示、Ver.1.1実験の起動。主な実行入口。 |
| `setup_windows.ps1` | Windows上で仮想環境と依存パッケージを構築するスクリプト。 |
| `source_snapshot/README.md` | スナップショットの実行入口、設定継承、実験完了状態の説明。 |
| `tests/test_cifar_vgg.py` | VGG16-BN構築と構造縮小のshape確認。 |
| `tests/test_compaction_state.py` | Gated/compact出力、構造メトリクス、momentumとscheduler移行を確認。 |
| `tests/test_evaluation.py` | 分類指標・FLOPs・パラメータ・モデルサイズ評価を確認。 |
| `tests/test_mask_distance.py` | Normalized Hamming mask距離を確認。 |
| `tests/test_mask_manager.py` | MaskManager/channel gate適用を確認。 |
| `tests/test_stability_detector.py` | Early-Bird安定判定と検出epochを確認。 |
| `tests/test_ts_controller.py` | t_s移行タイミング制御を確認。 |
| `training/__init__.py` | training package marker。 |
| `training/builders.py` | SGD optimizerとMultiStepLR schedulerを構築。 |
| `training/dense_training.py` | Dense epoch学習の薄いadapter。 |
| `training/optimizer_state.py` | 縮小モデルへのmomentum tensor slice移行とscheduler状態継承。 |
| `training/sparse_training.py` | MaskManagerを受け取るsparse epoch adapter。 |
| `training/trainer.py` | train epochとvalidation lossの基本routine。 |
| `utils/checkpoint.py` | State dict checkpointの保存補助。 |
| `utils/config.py` | YAML継承・設定merge・必須key検査。 |
| `utils/device.py` | CPU/CUDA device選択補助。 |
| `utils/logger.py` | run用logging補助。 |
| `utils/seed.py` | Python/NumPy/PyTorch乱数seedと決定論設定。 |

## 収録範囲

Pythonソース、全YAML設定、テスト、研究計画書、事前計画JSON、環境セットアップ用スクリプトを収録しています。CIFAR-10の画像データ、仮想環境、モデルweight checkpointはサイズ・再配布上の理由で収録していません。run条件、環境、mask、epoch記録、評価checkpointの最終サイズ・指標は`results/20261030T183030009791/`にあります。

実験コード本体の実行記録は`results/20261030T183030009791/runs/<experiment_id>/run.log`と`metrics.csv`、実際に使われた統合configは各runの`config.yaml`です。
