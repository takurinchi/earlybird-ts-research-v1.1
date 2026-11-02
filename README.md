# Early-Bird Ver.1.1 実験結果

CIFAR-10 / VGG16-BNで、Dense対照と、学習途中にモデルを物理縮小する方式を比較した実験記録です。
結果は各条件ごとに **Dense対照 → 物理縮小** の順で並べています。

## 実験条件

- Early-Bird baseline: `20261026T130454_baseline_seed1`、検出時点 `t_m=28`
- 条件: C1=`t_s=18`、C2=`t_s=28`、C3=`t_s=38`、C4=`t_s=40`
- 情報方式: `online`（run自身の`t_s`時点のBNマスク）と`posthoc`（baselineの`t_m`マスク）
- 各条件でDense対照と物理縮小を対にして比較。seed=1、160 epoch、BNチャネル選択率0.5
- 検証は毎epoch、テスト評価は学習完了後に1回
- 実行環境: Python 3.12.14、PyTorch 2.14.1+cu130、CUDA 13.0、NVIDIA GeForce RTX 5050

## 条件別結果

速度比はDense対照 ÷ 物理縮小です。速度比1.2倍は、所要時間が約16.7%短いことを意味します。

| 情報方式 | 条件 | t_s | 学習状態 | 最終テスト精度 | t_s後学習秒 | run全体秒 | パラメータ削減率 | FLOPs削減率 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| online | C1 (t_m−10) | 18 | Dense対照（マスク後もDense） | 92.42% | 2863.0 | 3394.8 | 78.91% | 56.23% |
| online | C1 (t_m−10) | 18 | 物理縮小（t_s+1から） | 92.74% | 2296.0 | 2781.2 | 78.91% | 56.23% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.247× | 1.221× |  |  |
| online | C2 (t_m) | 28 | Dense対照（マスク後もDense） | 92.47% | 2657.5 | 3354.0 | 81.42% | 42.66% |
| online | C2 (t_m) | 28 | 物理縮小（t_s+1から） | 92.89% | 2114.4 | 2791.8 | 81.42% | 42.66% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.257× | 1.201× |  |  |
| online | C3 (t_m+10) | 38 | Dense対照（マスク後もDense） | 92.54% | 2454.8 | 3345.4 | 82.03% | 38.69% |
| online | C3 (t_m+10) | 38 | 物理縮小（t_s+1から） | 92.93% | 2027.0 | 2909.2 | 82.03% | 38.69% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.211× | 1.150× |  |  |
| online | C4 (固定) | 40 | Dense対照（マスク後もDense） | 92.81% | 2416.6 | 3346.5 | 81.89% | 37.73% |
| online | C4 (固定) | 40 | 物理縮小（t_s+1から） | 92.67% | 2009.5 | 2931.4 | 81.89% | 37.73% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.203× | 1.142× |  |  |
| posthoc | C1 (t_m−10) | 18 | Dense対照（マスク後もDense） | 92.29% | 2856.6 | 3358.5 | 81.67% | 43.38% |
| posthoc | C1 (t_m−10) | 18 | 物理縮小（t_s+1から） | 92.09% | 2255.7 | 2735.9 | 81.67% | 43.38% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.266× | 1.228× |  |  |
| posthoc | C2 (t_m) | 28 | Dense対照（マスク後もDense） | 92.62% | 2658.5 | 3355.3 | 81.67% | 43.38% |
| posthoc | C2 (t_m) | 28 | 物理縮小（t_s+1から） | 92.41% | 2095.7 | 2772.0 | 81.67% | 43.38% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.269× | 1.210× |  |  |
| posthoc | C3 (t_m+10) | 38 | Dense対照（マスク後もDense） | 92.66% | 2459.8 | 3353.0 | 81.67% | 43.38% |
| posthoc | C3 (t_m+10) | 38 | 物理縮小（t_s+1から） | 92.40% | 1938.0 | 2810.0 | 81.67% | 43.38% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.269× | 1.193× |  |  |
| posthoc | C4 (固定) | 40 | Dense対照（マスク後もDense） | 92.86% | 2416.8 | 3347.2 | 81.67% | 43.38% |
| posthoc | C4 (固定) | 40 | 物理縮小（t_s+1から） | 92.74% | 1916.7 | 2831.5 | 81.67% | 43.38% |
|  | **ペア速度比** |  | Dense ÷ 物理縮小 |  | 1.261× | 1.182× |  |  |

`t_s後学習秒`はDense対照ではマスク後のDense学習、処置runでは物理縮小後の学習を表します。`run全体秒`はrun起動から最終評価・checkpoint保存までで、結果bundle生成時間は含みません。

### 結果ファイル

- [対比較CSV](results/20261030T183030009791/compaction_comparison.csv): 8組を1行ずつ掲載
- [run別summary.csv](results/20261030T183030009791/summary.csv): Dense対照→物理縮小の順で16 runを掲載
- [全epoch CSV](results/20261030T183030009791/metrics_all_epochs.csv): 同じ順でrunごとの160 epochを掲載
- [run順序インデックス](results/20261030T183030009791/RUN_INDEX.csv)
- [精度plot](results/20261030T183030009791/plots/accuracy_by_ts.png)、[学習時間plot](results/20261030T183030009791/plots/training_time_by_ts.png)
- `results/20261030T183030009791/runs/`にはrun単位の設定、環境、ログ、CSV、summary、遷移マスクを収録

## ソースと再現情報

- [ソースコード一覧・各ファイルの役割](SOURCE_CODE.md)
- [`source_snapshot/`](source_snapshot/)に実験時コード、全設定、計画書、テストを収録
- ソーススナップショットの入口・依存関係は[`source_snapshot/README.md`](source_snapshot/README.md)
- 大容量の学習済み重みcheckpointとCIFAR-10データ本体は含めていません。baselineの設定・マスク・記録はbundle内に保存しています。

## 再実行

Python環境とCIFAR-10データを準備し、`source_snapshot`を作業ディレクトリとして実行します。

```powershell
python run_experiments.py --config configs/ts_compaction.yaml --execute --resume-baseline ..\results\20261030T183030009791\baseline
```
