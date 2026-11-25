# Milestone 3 本番規模ラン実行手順

## 設定

使用config: `configs/two_stage_m3_full.yaml`

- データ: `../kuzushiji-recognition/char_sep_datas` 配下の全クラス・全画像。クラス数・画像数の上限なし。
- split: `unseen_character`。文字単位にtrain/validation/testへ分け、文字集合が互いに重ならない。比率は0.8/0.1/0.1、seed 42。
- Stage 1: skeleton classifier、最大100 epoch、validation skeleton accuracyでbest checkpointを選び、10 epoch連続で改善しなければ停止。
- Stage 2: ViT `filled_marker` oracle、ImageNet ViT-B/16事前学習重みあり、最大100 epoch、validation slot accuracyでbest checkpointを選び、10 epoch連続で改善しなければ停止。
- 評価: test画像でStage 1 skeleton accuracy、正解スケルトンを入力するStage 2 oracleのslot accuracy、slot exact match、文字精度、辞書hit率、およびdepth0/depth1/depth2plus別の値を記録。
- W&B: `zero-shot-kuzushiji` projectへepochごとのtrain/validation lossとStage 1 skeleton accuracy、Stage 2 slot accuracy/exact matchを記録。認証情報は既存のW&B login設定を使用し、APIキーをconfigやコマンドへ書かない。
- 生成物: `outputs/261125_two_stage_m3_full/unseen_character/` 以下へcheckpoint、summary、`evaluation.json`、labnote result YAMLを保存。W&Bのローカルログも`outputs/261125_two_stage_m3_full/wandb/`以下。

Stage 1とStage 2は別プロセス・別モデルです。以下をPowerShellで上から順に実行してください。各コマンドはリポジトリrootから起動し、表示内容を同時に専用フォルダのlogへ保存します。

## 実行コマンド

```bat
Set-Location "C:\Users\kotat\MyPrograms\MyKuzushiji\zero-shot-kuzushiji"
conda activate kuzushiji
New-Item -ItemType Directory -Force -Path outputs/261125_two_stage_m3_full/unseen_character, outputs/261125_two_stage_m3_full/wandb | Out-Null

python -m src.two_stage.train_stage1 --config configs/two_stage_m3_full.yaml 2>&1 | Tee-Object -FilePath outputs/261125_two_stage_m3_full/unseen_character/stage1_train.log
python -m src.two_stage.train_stage2 --config configs/two_stage_m3_full.yaml 2>&1 | Tee-Object -FilePath outputs/261125_two_stage_m3_full/unseen_character/stage2_train.log
python -m src.two_stage.evaluate --config configs/two_stage_m3_full.yaml 2>&1 | Tee-Object -FilePath outputs/261125_two_stage_m3_full/unseen_character/evaluate.log
```

Stage 2のImageNet ViT-B/16重みがtorchvision cacheにない場合は、初回起動時にtorchvisionが公式重みをダウンロードします。W&Bへオンライン記録するため、実行環境で既存のW&B login状態とネットワーク接続を利用してください。

評価YAMLは`outputs/261125_two_stage_m3_full/unseen_character/result_two-stage-m3-full-unseen-261125.yaml`、詳細JSONは同じディレクトリの`evaluation.json`です。Stage 1/Stage 2の最良checkpointはそれぞれ`stage1/best.pt`、`stage2/best.pt`に保存されます。
