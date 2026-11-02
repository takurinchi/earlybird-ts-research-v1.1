# 研究計画書・実装仕様書 Ver.1.1（学習中モデル縮小の対照実験）

## 1. 研究概要

### 1.1 研究テーマ

Early-Birdによるマスク安定化時点 \(t_m\) とスパース化移行時点 \(t_s\) の比較に、\(t_s\) 到達後にモデルを物理的に縮小して学習を続ける条件を追加する。従来方式（Dense構造のまま学習を完了し、学習後に小型モデルを作成する条件）と比較し、学習時間・精度・縮小処理コストへの影響を測定する。

### 1.2 背景と現状の課題

Ver.1の実装では、\(t_s\) 到達時にBatch Normalization（BN）の出力へマスクを適用する。その後もDense構造のモデルで全160 epochを学習し、物理的な構造削減は学習ループ終了後に行う。このため、実験結果のパラメータ数やFLOPsは削減されても、学習中にチャネル演算がなくならず、\(t_s\) の違いによる学習時間短縮が生じにくい。

Ver.1.1では、\(t_s\) でマスクに対応するチャネルを物理的に削除し、縮小後のモデルを残りepochで学習する実験群を追加する。これにより、学習中にモデルを縮小することが時間短縮につながるかを、従来方式と対照して調べる。

### 1.3 研究目的・問い

1. \(t_s\) で物理的にモデルを縮小すると、\(t_s\) 以降の学習時間と総学習時間は短縮するか。
2. 縮小処理とoptimizer状態の移行に要するコストを含めても、総実行時間に利点があるか。
3. 物理的縮小が最終Accuracy・Macro-F1に与える影響は、従来方式と比べてどの程度か。
4. \(t_m\) と \(t_s\) の位置関係が、時間短縮と精度のトレードオフにどう影響するか。

主仮説は「物理的縮小後の学習時間は短くなるが、総時間の短縮幅は \(t_s\)、縮小処理時間、GPU上の実効性能に依存する」とする。精度について方向を仮定せず、実測で比較する。

### 1.4 研究範囲

**追加する内容**

- Ver.1のDense対照方式に加え、\(t_s\) 到達時にモデルを物理的に縮小する方式
- C1〜C4、online/posthocの各組合せで方式間を比較
- optimizerのmomentum状態を縮小モデルへ引き継ぐ処理
- \(t_s\) 前後、縮小処理、評価処理を分けた時間測定
- 縮小直後のモデル整合性確認と、epochごとのモデル規模記録

**対象外**

- Rewinding時点 \(t_r\) の比較
- 新しいモデル・データセット・最適化手法・枝刈り率の探索
- 量子化、蒸留、非構造化枝刈りなど他の高速化方式
- 本計画書作成時点での実験実行（実装後の検証・許可なく本実験を開始しない）

---

## 2. Ver.1から引き継ぐ条件

既存コードと結果バンドルで確認できる条件を引き継ぐ。実装開始時に、設定ファイルと保存されたrun設定を再確認する。

| 項目 | 引き継ぐ値・仕様 |
|---|---|
| データセット | CIFAR-10、既存のtrain/validation分割とaugmentation |
| モデル | CIFAR VGG16-BN |
| seed | まず既存比較と同じ1。一般化を主張する場合は追加seedを設ける |
| 学習epoch | 160 |
| batch size | 256 |
| optimizer | SGD、learning rate 0.1、momentum 0.9、weight decay 0.0001 |
| LR schedule | 80・120 epochでgamma 0.1のMultiStepLR |
| Early-Bird条件 | epsilon 0.1、mask window 5、既存の候補マスク・距離実装 |
| 選択チャネル比 | 既存設定 `selected_ratio: 0.5` |
| 検出run | 保存済みDense runは \(t_m=28\)。再利用時はseed・設定・マスクの対応を確認 |
| C1 | \(t_s=t_m-10=18\) |
| C2 | \(t_s=t_m=28\) |
| C3 | \(t_s=t_m+10=38\) |
| C4 | 固定 \(t_s=40\) |
| 情報方式 | onlineとposthocを別条件として記録 |

既存結果の検出runは `outputs/results/online_c4_20261026/runs/baseline` に集約されている。posthocの安定マスクを使う場合は、そのマスクファイル、生成run ID、\(t_m\)、ハッシュ値を新しい各runのメタデータへ記録する。

---

## 3. 用語・実験方式

| 名称 | 定義 |
|---|---|
| Dense対照（late compaction） | \(t_s\) でBN出力マスクを適用するがDense構造で160 epochまで学習し、学習後に評価用モデルを物理縮小する。Ver.1方式 |
| 学習中縮小（train-time compaction） | \(t_s\) の直後、残りepochの学習前にモデルを物理縮小し、縮小モデルで学習を続ける |
| online | \(t_s\) 時点のBN重要度からその場でマスクを生成。未来の安定マスクを参照しない |
| posthoc | 検出runで得た \(t_m\) の安定マスクを使用。onlineと別の情報方式として比較する |
| 縮小直前checkpoint | \(t_s\) まで学習したDenseモデルの重み・BN統計・optimizer状態 |
| compaction time | Denseモデルから縮小モデルを生成し、optimizer状態を移行する処理時間 |

### 3.1 移行epochの定義

現行コードの意味を維持し、epoch \(t_s\) の学習を完了した直後、epoch \(t_s+1\) の学習開始前に移行する。例えばC2は28 epochをDenseで完了し、epoch 29から縮小モデルで学習する。CSV・ログには `transition_after_epoch=t_s` と `first_compact_epoch=t_s+1` の両方を記録する。

### 3.2 比較条件

| 因子 | 水準 |
|---|---|
| 学習方式 | Dense対照、学習中縮小 |
| 情報方式 | online、posthoc |
| 移行時点 | C1=18、C2=28、C3=38、C4=40（\(t_m=28\) の場合） |

全組合せは2 × 2 × 4 = 16 run（Dense対照8、学習中縮小8）。マスク検出用Dense runは共有元として識別し、比較runとして二重計上しない。既存Ver.1結果は参考比較として残すが、厳密な対照には同じコード版・seed・データ順で実行した対応Dense対照runを用いる。

---

## 4. 学習中縮小の実装仕様

### 4.1 移行処理

1. Dense対照と同じ初期重み・データ順で学習する。
2. \(t_s\) の学習epochを完了する。
3. onlineでは現在時点のBN重要度からマスクを作る。posthocでは固定された \(t_m\) マスクを読み込む。
4. 現行 `structurally_prune_vgg16` と同じチャネル選択規則でモデルを物理縮小し、Conv、BN、分類器の重みとBN running statisticsをコピーする。
5. Dense側のマスクhookを解除し、縮小モデルへoptimizerを接続し直す。
6. SGD momentum bufferを、各Conv/BN/Linear parameterの生存チャネルindexに従ってsliceし、新モデルの対応parameterへ移す。momentumの対象外となるscalar状態は規則を定めて移行する。
7. schedulerの現在epochとlearning rateを維持し、80・120 epochのmilestoneが元の絶対epochで適用されるようにする。
8. epoch \(t_s+1\) からepoch 160まで縮小モデルを学習する。

optimizer stateを正しく対応付けられない場合、黙って初期化しない。実装段階で移行の正当性を検証し、困難な場合は「momentum reset」方式を別の実験条件として明示し、主比較に混在させない。

### 4.2 数値的一貫性

本実験の前に短い検証を行い、同じDense重み・同じマスクについて、BN出力マスクを適用したモデルと物理縮小モデルのlogitsが許容誤差内で一致することを確認する。これは構造変換の検証であり、性能実験runとは分けて記録する。

縮小によって残すweight、BN affine parameter、running mean/variance、classifier入力列、およびmomentum bufferが同一のchannel indexを使っていることを検証する。最初のcompact epoch前に、全parameter数・各層のchannel数・optimizer param group・learning rateをログへ保存する。

### 4.3 公平性の条件

- 対応するDense対照と学習中縮小runでseed、初期重み、訓練/検証分割、augmentation、batch順序、mask、\(t_s\)、総epoch数、optimizer設定を揃える。
- 各run開始時にseedを設定し、train DataLoaderのshuffle順も条件間で再現可能にする。
- 同じ情報方式内でのみマスクの由来を揃えて方式比較する。
- schedulerはglobal epoch基準とし、縮小時にスケジュールを最初からやり直さない。
- 縮小処理時間は学習時間から隠さず別記録し、end-to-end時間にも含める。
- 初回比較は既存seed=1で行う。統計的一般化を結論づける前に複数seedで反復し、平均・標準偏差を報告する。

### 4.4 ログ・検証・checkpointの頻度

比較方式によるI/Oや検証処理の差が測定時間へ混ざらないよう、対応するDense対照と学習中縮小runで次の頻度を揃える。

- **検証**：全160 epochで毎epoch実施する。使用データ、評価関数、batch sizeを共通にし、最終テスト評価は全runで学習終了後に1回実施する。
- **epochログ**：各epochにつき1件を`metrics.csv`へ記録し、`run.log`にも同じ頻度で進捗を記録する。検証値、epoch学習時間、現在のlearning rate、現在のparameter countを含める。
- **移行ログ**：全runでepoch \(t_s\) 完了時に同じ形式のtransition eventを記録する。学習中縮小runは追加でcompact modelのshape、optimizer/scheduler移行結果、compaction時間を記録する。
- **checkpoint**：比較runでは初期状態と最終状態を各1回保存する。移行時checkpointを有効にする場合は、Dense対照と学習中縮小runの両方でepoch \(t_s\) の同じ境界に対応するcheckpointを保存する。方式片側だけの定期保存は行わない。
- **検出用run**：マスク検出run固有の`tm.pt`、mask-distance CSV/plotは検出成果物として保存し、比較runのcheckpoint頻度・実行時間との直接比較には混ぜない。
- **時間計測**：epoch学習時間、検証時間、checkpoint保存時間、compaction時間を別に計測する。方式間比較に使うend-to-end時間には実際に発生した全処理を含め、学習カーネル時間だけの比較とは分けて報告する。

移行時checkpointを保存しない設定の場合も、その設定を全run共通にし、初期・最終checkpointだけを保存する。checkpointの有無・保存範囲はconfigとmanifestへ記録する。

---

## 5. 評価指標と時間計測

### 5.1 性能・モデル指標

Accuracy、Macro-F1、loss、parameter count、FLOPs/sample、model size、推論時間をVer.1と同じ評価定義で記録する。学習中縮小方式では、Dense開始時、縮小直前、縮小直後、最終時点のモデル指標を保存する。

### 5.2 時間指標

| 指標 | 定義 |
|---|---|
| `dense_train_seconds` | epoch 1から \(t_s\) までのepoch内学習時間合計 |
| `compact_train_seconds` | \(t_s+1\) から160までのepoch内学習時間合計。Dense対照ではmasked Dense区間の時間 |
| `compaction_seconds` | 物理縮小とoptimizer/scheduler状態移行に要した時間。Dense対照は0または適用なし |
| `validation_seconds` | 全epochの検証時間合計 |
| `training_seconds` | 全epochの学習時間合計。従来との比較用 |
| `wall_seconds` | run開始から学習・検証ループ完了まで。既存定義を維持 |
| `end_to_end_seconds` | 初期化後から最終評価・保存まで。方式の総コスト比較用 |
| `speedup_after_ts` | 対応Dense区間の `compact_train_seconds / dense_train_seconds` 比ではなく、Dense対照の \(t_s\) 後時間を学習中縮小の \(t_s\) 後時間で割る |
| `total_speedup` | 対応Dense対照のend-to-end時間 ÷ 学習中縮小のend-to-end時間 |

GPU計測ではepoch境界でCUDA synchronizeしてから計測し、方式間で同じ計測範囲を使う。`training_seconds` と `wall_seconds` を混同せず、データ読み込み、検証、縮小、checkpoint保存を別項目またはend-to-endに反映する。

---

## 6. 保存成果物とディレクトリ案

既存のrun単位出力を保ち、比較結果を別の実験bundleに保存する。既存Ver.1 bundleを上書きしない。

```text
configs/
  ts_compaction.yaml                 # 新しい方式因子と保存設定
outputs/
  <run_id>_<condition>/              # 既存のrun形式を再利用
    config.yaml
    environment.json
    run.log
    metrics.csv
    summary.json
    masks/
      transition_mask.pt
    checkpoints/
      pre_compaction.pt               # 任意設定。容量方針に従う
      final.pt
outputs/results/
  train_time_compaction_v1_1/
    README.md
    manifest.json
    summary.csv
    metrics_all_epochs.csv
    compaction_comparison.csv
    plots/
      training_time_by_ts.png
      accuracy_by_ts.png
```

`metrics.csv` / `summary.json` に最低限、`training_strategy`、`information_mode`、`ts`、`tm`、`seed`、`transition_after_epoch`、`first_compact_epoch`、`parameter_count_before_compaction`、`parameter_count_after_compaction`、前節の時間項目、最終性能、削減率、完了状態、mask source run IDとhashを保存する。

Checkpoint保存は容量が大きいため設定可能にする。少なくとも初期状態、縮小直前状態、最終状態の保存有無と対象をmanifestに明記し、結果CSV・設定・ログ・mask・要約は再解析できる形で管理する。

---

## 7. 現行ディレクトリ構成との対応

| 既存ファイル | Ver.1.1での役割・変更候補 |
|---|---|
| `cifar_experiment.py` | `t_s` eventで方式分岐し、物理縮小、状態移行、区間時間計測、metadata保存を制御 |
| `models/cifar_vgg.py` | `structurally_prune_vgg16`を拡張し、channel index mappingも返せるようにする |
| `pruning/channel_manager.py` | Dense対照のactivation maskを管理。物理縮小後はhookを確実に解除 |
| `training/builders.py` | compact model用optimizer再構築を補助。既存hyperparameterを維持 |
| `training/trainer.py` | Dense/compact両方で同じ学習処理を利用。実時間区間の外側計測を明確化 |
| `data/cifar10.py` | split・augmentationを維持し、matched runのsample順を再現可能にする |
| `configs/ts_sweep.yaml` | 既存Ver.1条件を維持 |
| 新規 `configs/ts_compaction.yaml` | strategy因子、条件、optimizer state移行、checkpoint、時間計測を設定 |
| `analysis/aggregate_results.py` | 新しいrun metadataと時間列を欠落なく集計 |
| `analysis/plot_ts_comparison.py` | 必要に応じ方式別の精度・時間比較plotを追加 |
| `analysis/package_posthoc_results.py` | Ver.1の結果bundle作成用。Ver.1.1 bundleを上書きしないよう別処理または汎用化 |
| `tests/test_cifar_vgg.py` | compact modelのshape、weight、BN統計、logits整合性の検証 |
| `tests/test_mask_manager.py` | hook適用・解除とDense対照のマスク動作を検証 |
| 新規 `tests/test_compaction_state.py` | optimizer state mapping、scheduler epoch/LR維持、遷移epochを検証 |

上記は実装方針であり、本計画書作成時点ではコード変更・実験実行を行わない。

---

## 8. 検証・実験手順

### 8.1 実装検証

1. 既存Ver.1実装・設定・出力形式を確認し、変更前のcommitを記録する。
2. 構造縮小時に必要なsurviving channel index mappingを設計する。
3. compact modelのforward一致、parameter数、optimizer momentum移行、scheduler継続を小規模検証する。
4. \(t_s\) の直前・直後に一度だけ遷移し、epoch重複・欠落がないことを確認する。
5. 既存Dense対照方式が従来どおり動作することを確認する。
6. 少数batch/短epochのsmoke runでartifactと時間列を確認する。

### 8.2 本実験

1. 設定・コードrevision・環境・mask sourceを固定してmanifestを作成する。
2. 必要なDense detector runとマスクを確認する。
3. 16条件をC1からC4まで、Dense対照と学習中縮小で対応付けて実行する。GPU負荷の偏りが比較に影響する場合は実行順を交互またはランダム化して記録する。
4. 各run終了時にCSV/JSON/ログ/checkpointの整合性を確認する。
5. 対応条件の精度、時間、構造指標を集計し、方式・\(t_s\) 別に可視化する。
6. 単一seedの結果は予備結果として扱い、複数seedの結果が揃うまでは一般化を限定する。

---

## 9. 解釈上の注意

- チャネル削減率と実測時間短縮率は一致しない。GPU kernel、batch size、データ供給、固定コストの影響を受ける。
- `training_seconds`だけでは方式の実用コストを評価できない。`compaction_seconds`を含むend-to-end時間を併記する。
- 学習途中でoptimizer stateをリセットすると方式差とoptimizer差が交絡する。主実験では可能な限りmomentum stateを移行する。
- 既存Ver.1の結果と新規runは、コード版やmask sourceが異なる可能性がある。厳密比較は同一実験計画で再実行したmatched Dense対照を使う。
- 同一seed一回の差は乱数変動を含む。時間・精度の結論には複数seed反復が望ましい。
- 学習中に構造縮小しても、データロードや検証にかかる時間は残る。実時間の有意な短縮を事前に保証しない。

---

## 10. 実施判断

本書はVer.1.1の実験計画と実装仕様を定める。コード実装後は、本書の方式分岐、状態移行、時間計測、出力仕様を検証してから本実験へ進む。Ver.1結果bundleを保持したまま新しいbundleへ保存する。
