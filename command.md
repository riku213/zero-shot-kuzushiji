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