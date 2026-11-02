# Early-Bird Ver.1.1 実験結果

CIFAR-10 / VGG16-BNで、学習中にモデルを物理縮小する方式と、Dense構造のまま学習を継続する対照方式を比較した実験記録です。

## 実験条件

- データセット・モデル: CIFAR-10、VGG16-BN
- Early-Bird検出baseline: `20261026T130454_baseline_seed1`、`t_m=28`
- 比較: `dense_late_compaction`（Dense対照）と`train_time_compaction`（`t_s`到達後に物理縮小）
- マスク: `online`（各runの`t_s`時点で生成）、`posthoc`（baselineの`t_m=28`マスク）
- `t_s`: C1=18、C2=28、C3=38、C4=40。各条件2方式を対にして、計16 runを実行
- seed=1、160 epoch、選択削減率=0.5。検証は毎epoch、テスト評価は学習後に1回
- 実行環境: Python 3.12.14、PyTorch 2.14.1+cu130、CUDA 13.0、NVIDIA GeForce RTX 5050

## 条件別結果

| 条件 | マスク | t_s | t_s後の速度比 | 全体速度比 | Dense精度 | 物理縮小精度 |
|---|---|---:|---:|---:|---:|---:|
| C1 (t_m−10) | online | 18 | 1.247× | 1.221× | 92.42% | 92.74% |
| C2 (t_m) | online | 28 | 1.257× | 1.201× | 92.47% | 92.89% |
| C3 (t_m+10) | online | 38 | 1.211× | 1.150× | 92.54% | 92.93% |
| C4 (固定) | online | 40 | 1.203× | 1.142× | 92.81% | 92.67% |
| C1 (t_m−10) | posthoc | 18 | 1.266× | 1.228× | 92.29% | 92.09% |
| C2 (t_m) | posthoc | 28 | 1.269× | 1.210× | 92.62% | 92.41% |
| C3 (t_m+10) | posthoc | 38 | 1.269× | 1.193× | 92.66% | 92.40% |
| C4 (固定) | posthoc | 40 | 1.261× | 1.182× | 92.86% | 92.74% |

速度比はDense対照時間 ÷ 物理縮小方式時間です。値が1を超えると物理縮小方式のほうが短時間です。精度差は条件ごとに異なります。

## 主な成果物

- [比較CSV](results/20261030T183030009791/compaction_comparison.csv)
- [全epoch指標CSV](results/20261030T183030009791/metrics_all_epochs.csv)
- [条件別summary](results/20261030T183030009791/summary.csv)
- [実験manifest](results/20261030T183030009791/manifest.json)
- [精度plot](results/20261030T183030009791/plots/accuracy_by_ts.png)
- [学習時間plot](results/20261030T183030009791/plots/training_time_by_ts.png)
- `results/20261030T183030009791/runs/`に16 run分のconfig、environment、metrics、log、summary、遷移マスクを収録
- `source_snapshot/`に実験時のコード・設定・テストを収録

大容量の学習済みweight checkpointとCIFAR-10本体はこのリポジトリには含めていません。各runのepoch指標、ログ、マスク、summaryは収録しています。

## 再実行

Python環境とCIFAR-10データを用意した後、`source_snapshot`を作業ディレクトリにして実行します。

```powershell
python run_experiments.py --config configs/ts_compaction.yaml --execute --resume-baseline ..\results\20261030T183030009791\baseline
```

再実行時は`source_snapshot/configs/earlybird.yaml`と`ts_compaction.yaml`に記載された条件が使われます。
