import zipfile
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ==========================================
# 設定
# ==========================================
# ZIP_PATH = r"C:\Downloads\casia-hwdb-dataset.zip"  # ダウンロードしたZIPのパス
ZIP_PATH = r"C:\Users\kotat\Downloads\archive.zip"  # ダウンロードしたZIPのパス
# EXTRACT_DIR = r"C:\path\to\your\project\dataset\casia_hwdb" # 直接展開したい目的のパス
EXTRACT_DIR = r"C:\Users\kotat\MyPrograms\MyKuzushiji\kuzushiji-recognition\CASIA-HWDB" # 直接展開したい目的のパス
# ==========================================

def extract_file(zip_path, member_name, extract_to):
    """個別のファイルを展開するワーカー関数"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract(member_name, path=extract_to)

def main():
    zip_path = Path(ZIP_PATH)
    extract_dir = Path(EXTRACT_DIR)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("ZIPファイル内のファイルリストを取得中...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # ディレクトリ以外の実ファイルだけを抽出対象にする
        members = [m.filename for m in zf.infolist() if not m.is_dir()]

    total_files = len(members)
    print(f"合計 {total_files} 個のファイルを展開します...")
    print(f"展開先: {extract_dir}")

    # ProcessPoolExecutorを使って全CPUコアで並列解凍
    # （※メモリとCPUを極限まで使います）
    extracted_count = 0
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(extract_file, zip_path, member, extract_dir)
            for member in members
        ]
        
        # 進捗の表示
        for i, future in enumerate(futures, 1):
            future.result() # エラーがあればここでキャッチ
            if i % 10000 == 0:
                print(f"進捗: {i} / {total_files} ファイル完了")

    print("展開が完了しました！")

if __name__ == '__main__':
    main()