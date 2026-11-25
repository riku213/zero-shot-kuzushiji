# 2026-11-25 Stage 1 + Stage 2 基準実験

## 概要

§8マイルストーン3の小規模確認後、標準configでStage 1スケルトン分類器とStage 2 `filled_marker` oracle補完器を学習・評価した。実行環境はconda `kuzushiji`（Python 3.12.3、CUDA、NVIDIA GeForce RTX 4090）。W&B接続先はForge CoreWeave。

Stage 1とStage 2は独立モデル・独立checkpoint。Stage 2評価は正解スケルトンを入力し、glyph codeおよびguidance/triplet補助損失は無効。IDS leaf解決に`outputs/261018_cos_median_codebook/all_glyph_codes.pkl`と`outputs/261007_correct_dict/combined_render_metadata.json`を使用した。

## 小規模ラン

| 項目 | 設定・結果 |
|---|---|
| Config | `configs/two_stage_m3_smoke.yaml` |
| データ | 12クラス、84画像。train 65、validation 7、test 12 |
| 学習 | Stage 1 1 epoch、Stage 2 1 epoch |
| Stage 1 validation | skeleton accuracy 14.29% |
| Stage 2 validation | slot loss 7.9493 |
| Test | skeleton 8.33%、slot accuracy 3.57%、slot exact match 0.00%、oracle文字精度0.00%、辞書hit率0.00% |
| 深さ別test件数 | depth0 0、depth1 10、depth2plus 2 |
| W&B run | [Stage 1](https://forge.coreweave.com/wandb/rikuo213-hosei-university/zero-shot-kuzushiji/runs/zsmck86b)、[Stage 2](https://forge.coreweave.com/wandb/rikuo213-hosei-university/zero-shot-kuzushiji/runs/z0hacj7w)、[評価](https://forge.coreweave.com/wandb/rikuo213-hosei-university/zero-shot-kuzushiji/runs/37uqm7t8) |
| 結果ファイル | `outputs/261125_two_stage_m3_smoke/result_two-stage-m3-smoke-261125.yaml` |
| 所要時間 | W&B runtime合計 約7秒（各runのSDK runtimeの合算。プロセス起動時間は別） |

小規模ランの目的であるStage 1/2学習、checkpoint保存、oracle評価、W&B記録とartifact同期、labnote YAML生成まで完了した。smoke runで検出したStage 2 DataLoader batch型とIDS深さ上限の空fallback問題を修正してから標準ランを実施した。

## 標準config本ラン

| 項目 | 設定・結果 |
|---|---|
| Config | `configs/two_stage_default.yaml` |
| データ | 256クラス、2,427画像、leaf vocabulary 10,016 |
| Split | train 1,821画像/256クラス、validation 268画像/159クラス、test 338画像/213クラス |
| Stage 1 | 5 epoch、best epoch 5、best validation skeleton accuracy 73.13%、test 74.26% |
| Stage 2 | 5 epoch、best epoch 5、best validation slot loss 3.4081 |
| W&B run | [Stage 1](https://forge.coreweave.com/wandb/rikuo213-hosei-university/zero-shot-kuzushiji/runs/l02kwyq7)、[Stage 2](https://forge.coreweave.com/wandb/rikuo213-hosei-university/zero-shot-kuzushiji/runs/l24spwpl)、[評価](https://forge.coreweave.com/wandb/rikuo213-hosei-university/zero-shot-kuzushiji/runs/gk8yb860) |
| 結果ファイル | `outputs/two_stage_baseline/evaluation.json`、`outputs/two_stage_baseline/result_two-stage-261125.yaml` |
| 所要時間 | W&B runtime: Stage 1 約20秒、Stage 2 約32秒、評価約2秒、計約54秒。プロセス・データ準備時間を含まない |

### §7 指標（test）

| 指標 | 全体 |
|---|---:|
| Stage 1 スケルトン精度 | 74.26% |
| Stage 2 スロット精度（oracle skeleton） | 17.73% |
| 部品列完全一致率 | 4.73% |
| Oracle文字精度 | 4.73% |
| 辞書ヒット率 | 17.46% |

| 深さ | test画像数 | スケルトン精度 | スロット精度 | 部品列完全一致率 | Oracle文字精度 | 辞書ヒット率 |
|---|---:|---:|---:|---:|---:|---:|
| depth0 | 34 | 64.71% | 17.65% | 17.65% | 17.65% | 100.00% |
| depth1 | 285 | 80.35% | 15.37% | 3.16% | 3.16% | 8.42% |
| depth2plus | 19 | 0.00% | 39.06% | 5.26% | 5.26% | 5.26% |

スロット精度は全スロットに対する正解率、部品列完全一致率は画像単位で全スロットが正解した割合。Stage 2の文字精度は正解スケルトンを使ったoracle条件の評価であり、Stage 1の予測を接続したend-to-end文字精度ではない。辞書hitは復元IDSが逆引き辞書に存在する率で、認識正解率とは異なる。

## 再現コマンド

```bat
conda activate kuzushiji
python -m src.two_stage.train_stage1 --config configs/two_stage_m3_smoke.yaml
python -m src.two_stage.train_stage2 --config configs/two_stage_m3_smoke.yaml
python -m src.two_stage.evaluate --config configs/two_stage_m3_smoke.yaml
python -m src.two_stage.train_stage1 --config configs/two_stage_default.yaml
python -m src.two_stage.train_stage2 --config configs/two_stage_default.yaml
python -m src.two_stage.evaluate --config configs/two_stage_default.yaml
```
