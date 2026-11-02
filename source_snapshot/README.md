# Ver.1.1 source snapshot

このディレクトリは実験run `20261030T183030009791`に使用したソース・設定スナップショットです。ファイルごとの説明は[`../SOURCE_CODE.md`](../SOURCE_CODE.md)、実行結果は[`../README.md`](../README.md)を参照してください。

## 実行入口

```powershell
python run_experiments.py --config configs/ts_compaction.yaml --execute --resume-baseline ..\results\20261030T183030009791\baseline
```

設定は`configs/ts_compaction.yaml`から`earlybird.yaml`、`default.yaml`を継承します。CIFAR-10データとrequirementsにある環境が必要です。

## 完了した実験

16 run、各160 epochが完了済みです。実際の値は`../results/20261030T183030009791/`のbundleを参照してください。`experiment_plan_v1_1.json`は実験前の計画確認JSONであり、完了状態のmanifestではありません。
