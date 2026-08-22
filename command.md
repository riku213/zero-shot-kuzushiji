.\.venv\Scripts\python.exe src\6_train_model.py ^
  --data-root "../kuzushiji-recognition/char_sep_datas" ^
  --codebook "outputs/final_codebook.pkl" ^
  --output-dir "outputs" ^
  --epochs 1 ^
  --batch-size 8 ^
  --train-ratio 0.8 ^
  --device cuda