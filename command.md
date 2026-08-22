[26/08/22] 論文通りの実装
くずしじデータセットを使って学習したが，全くロスは収束しなかった．
原因はおそらく漢字の多種多様さのために，クラス数が多いわりにサンプルが少ないこと．
漢字のクラスは4146クラスであるのに対し，
2157クラスが10サンプル未満．
3204クラスが50サンプル未満．
```
.\.venv\Scripts\python.exe src\6_train_model.py ^
  --data-root "..\kuzushiji-recognition\char_sep_datas" ^
  --codebook "outputs\final_codebook.pkl" ^
  --output-dir "outputs" ^
  --checkpoint-path "outputs\260822_first_trial.pth" ^
  --epochs 20 ^
  --batch-size 32 ^
  --train-ratio 0.8 ^
  --device cuda
```

