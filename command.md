[26/08/22] 論文通りの実装
くずしじデータセットを使って学習したが，全くロスは収束しなかった．
原因はおそらく漢字の多種多様さのために，クラス数が多いわりにサンプルが少ないこと．
漢字のクラスは4146クラスであるのに対し，
2157クラスが10サンプル未満．
3204クラスが50サンプル未満．
```
.\.venv\Scripts\python.exe src\6_train_model.py ^
  --data-root "..\kuzushiji-recognition\char_sep_datas" ^
  --codebook "outputs\260828_codebook\final_codebook.pkl" ^
  --output-dir "outputs" ^
  --checkpoint-path "outputs\260822_first_trial.pth" ^
  --epochs 20 ^
  --batch-size 32 ^
  --train-ratio 0.8 ^
  --device cuda
```

[26/08/23] 大規模データセットで事前学習してファインチューニングする 
```
python src/6_train_model.py ^
  --data-root "../kuzushiji-recognition/char_sep_datas" ^
  --codebook "outputs/260828_codebook/final_codebook.pkl" ^
  --pretrain-root "../kuzushiji-recognition/CASIA-HWDB" ^
  --checkpoint-path "outputs/260822_finetuning/finetuning_fare_model.pth" ^
  --pretrain-checkpoint-path "outputs/260822_finetuning/pretrain_fare_model.pth" ^
  --device cuda ^
  --epochs 20 ^
  --pretrain-epochs 5 ^
  --batch-size 32
```

[26/08/24] manifestで大規模データセットのスキャンを次回以降高速化する
まずmanifestを作成する

```
python src/6_train_model.py ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --build-manifest
```

作成したmanifestで高速に学習を進める
```
python src/6_train_model.py ^
  --data-root "../kuzushiji-recognition/char_sep_datas" ^
  --codebook "outputs/260828_codebook/final_codebook.pkl" ^
  --pretrain-root "../kuzushiji-recognition/CASIA-HWDB" ^
  --manifest-path "outputs/manifests/main_manifest.txt" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --checkpoint-path "outputs/260822_finetuning/finetuning_fare_model.pth" ^
  --pretrain-checkpoint-path "outputs/260822_finetuning/pretrain_fare_model.pth" ^
  --device cuda ^
  --epochs 20^
  --pretrain-epochs 2 ^
  --batch-size 32
```

事前学習とファインチューニングを実行
＜テスト＞
```
python src/6_train_model.py ^
  --data-root "../kuzushiji-recognition/char_sep_datas" ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --manifest-path "outputs/manifests/main_manifest.txt" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --checkpoint-path "outputs/260828_finetuning/finetuning_fare_model.pth" ^
  --pretrain-checkpoint-path "outputs/260828_finetuning/pretrain_fare_model.pth" ^
  --epochs 1 ^
  --pretrain-epochs 1 ^
  --pretrain-max-classes 100 ^
  --pretrain-max-samples-per-class 20 ^
  --max-classes 100 ^
  --max-samples-per-class 20 ^
  --batch-size 16 ^
  --device cpu
```

＜本番＞
```
python src/6_train_model.py ^
  --data-root "../kuzushiji-recognition/char_sep_datas" ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --manifest-path "outputs/manifests/main_manifest.txt" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --checkpoint-path "outputs/260828_finetuning/finetuning_fare_model.pth" ^
  --pretrain-checkpoint-path "outputs/260828_finetuning/pretrain_fare_model.pth" ^
  --epochs 20 ^
  --pretrain-epochs 5 ^
  --batch-size 32 ^
  --device cuda
```

<Char_sep_dataにある文字だけで学習しようとするとデータが足りなすぎるので，CASIAデータセットにある文字種のcode bookも作成>
```
python src/5_build_final_codebook.py ^
  --ids-files dataset/ids_text/ids.txt dataset/ids_text/ids-cdp.txt ^
  --radical-codes outputs/radical_codes.pkl ^
  --dataset-root "../kuzushiji-recognition/char_sep_datas" ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --output outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl
```
↑なんかうまくいかなかった
codebook作成
```
python src/5_build_final_codebook.py ^
  --ids-files dataset/ids_text/ids.txt dataset/ids_text/ids-cdp.txt ^
  --radical-codes outputs/radical_codes.pkl ^
  --dataset-root "../kuzushiji-recognition/char_sep_datas" ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --output outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl
```
PretrainManifest作成
```
python src/6_train_model.py ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --build-manifest
```

[26/09/05] CASIA pretrain の予測結果と seen / unseen 評価をレポートとして出力
```bash
python tools/export_pretrain_results.py ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --reference-codebook "outputs/260828_codebook/final_codebook.pkl" ^
  --prediction-codebook "outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl" ^
  --checkpoint-path "outputs/pretrain_best_fare_model.pth" ^
  --output-dir "outputs/260901_codebook_CASIA/results" ^
  --sample-count 100 ^
  --train-ratio 0.8 ^
  --batch-size 16 ^
  --chunk-size 4096 ^
  --device cuda
```

出力されるテキストは、`image_path` / `fare_code` / `predicted_unicode` / `predicted_char` / `true_unicode` / `true_char` / `group` の列を持つ。
`group` は `seen` か `unseen` で、seen は学習済みクラスからのホールドアウト、unseen は学習に含まれないクラスを表す。

データセットの文字種認識が壊れていたので再度学習．
CASIAデータセットのうち，学習データとテストデータに分ける
テストデータには，学習データに含まれないクラスの文字と，学習データに含まれているが，学習データには含まれていないサンプルデータが含まれるようにした．
```
python src/6_train_model.py ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --manifest-path "outputs/manifests/main_manifest.txt" ^
  --codebook "outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl" ^
  --output-dir "outputs/260913_redefine_train_data" ^
  --checkpoint-path "outputs/260913_redefine_train_data/best_fare_model.pth" ^
  --pretrain-checkpoint-path "outputs/260913_redefine_train_data/pretrain_best_fare_model.pth" ^
  --state-path "outputs/260913_redefine_train_data/training_state.pth" ^
  --pretrain-state-path "outputs/260913_redefine_train_data/pretrain_training_state.pth" ^
  --metadata-path "outputs/260913_redefine_train_data/run_metadata.json" ^
  --pretrain-split-manifest-train "outputs/260913_redefine_train_data/pretrain_train_manifest.txt" ^
  --pretrain-split-manifest-seen-test "outputs/260913_redefine_train_data/pretrain_seen_test_manifest.txt" ^
  --pretrain-split-manifest-unseen-test "outputs/260913_redefine_train_data/pretrain_unseen_test_manifest.txt" ^
  --pretrain-train-class-ratio 0.8 ^
  --pretrain-seen-train-ratio 0.8 ^
  --pretrain-epochs 1 ^
  --epochs 20 ^
  --batch-size 32 ^
  --device cuda
```

事前学習ヲ20エポックやる．
python src/6_train_model.py ^
  --pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
  --pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
  --manifest-path "outputs/manifests/main_manifest.txt" ^
  --codebook "outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl" ^
  --output-dir "outputs/260913_redefine_train_data_fresh" ^
  --checkpoint-path "outputs/260913_redefine_train_data_fresh/best_fare_model.pth" ^
  --pretrain-checkpoint-path "outputs/260913_redefine_train_data_fresh/pretrain_best_fare_model.pth" ^
  --state-path "outputs/260913_redefine_train_data_fresh/training_state.pth" ^
  --pretrain-state-path "outputs/260913_redefine_train_data_fresh/pretrain_training_state.pth" ^
  --metadata-path "outputs/260913_redefine_train_data_fresh/run_metadata.json" ^
  --pretrain-split-manifest-train "outputs/260913_redefine_train_data_fresh/pretrain_train_manifest.txt" ^
  --pretrain-split-manifest-seen-test "outputs/260915_redefine_train_data_fresh/pretrain_seen_test_manifest.txt" ^
  --pretrain-split-manifest-unseen-test "outputs/260915_redefine_train_data_fresh/pretrain_unseen_test_manifest.txt" ^
  --pretrain-train-class-ratio 0.8 ^
  --pretrain-seen-train-ratio 0.8 ^
  --pretrain-epochs 20 ^
  --epochs 20 ^
  --batch-size 32 ^
  --device cuda

  学習再開コマンド

python src/6_train_model.py ^
--pretrain-root "C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/CASIA-HWDB" ^
--pretrain-manifest-path "outputs/manifests/pretrain_manifest.txt" ^
--manifest-path "outputs/manifests/main_manifest.txt" ^
--codebook "outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl" ^
--output-dir "outputs/260913_redefine_train_data_fresh" ^
--checkpoint-path "outputs/260913_redefine_train_data_fresh/best_fare_model.pth" ^
--pretrain-checkpoint-path "outputs/260913_redefine_train_data_fresh/pretrain_best_fare_model.pth" ^
--state-path "outputs/260913_redefine_train_data_fresh/training_state.pth" ^
--pretrain-state-path "outputs/260913_redefine_train_data_fresh/pretrain_training_state.pth" ^
--metadata-path "outputs/260913_redefine_train_data_fresh/run_metadata.json" ^
--pretrain-split-manifest-train "outputs/260913_redefine_train_data_fresh/pretrain_train_manifest.txt" ^
--pretrain-split-manifest-seen-test "outputs/260913_redefine_train_data_fresh/pretrain_seen_test_manifest.txt" ^
--pretrain-split-manifest-unseen-test "outputs/260913_redefine_train_data_fresh/pretrain_unseen_test_manifest.txt" ^
--pretrain-train-class-ratio 0.8 ^
--pretrain-seen-train-ratio 0.8 ^
--pretrain-epochs 20 ^
--epochs 20 ^
--batch-size 32 ^
--device cuda ^
--log-path "outputs/260913_redefine_train_data_fresh/train_resume.log"