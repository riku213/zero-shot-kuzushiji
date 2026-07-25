import urllib.request

# 1. データの読み込み（前回と同様）
url_standard = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"
url_cdp = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids-cdp.txt"

def load_extended_ids(urls):
    ids_dict = {}
    for url in urls:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            for line in response.read().decode('utf-8').splitlines():
                if line.startswith(';'): 
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ids_dict[parts[1]] = parts[2]
    return ids_dict

print("データをダウンロードしています...")
chise_dict = load_extended_ids([url_standard, url_cdp])
print(f"読み込み完了！\n")

# 2. 検索用の汎用関数
def search_variants(ids_dict, required_components):
    """
    指定した構成部品をすべて含む文字を辞書から検索する関数
    
    required_components: 検索したい部品のリスト
        例: ["山", "鳥"] -> 山と鳥を両方含む文字
        例: [("艹", "艸"), "月"] -> (艹または艸) と 月 を含む文字
    """
    results = []
    for char, ids_str in ids_dict.items():
        match = True
        for comp in required_components:
            # タプルやリストの場合は「OR条件」として処理
            if isinstance(comp, (tuple, list)):
                if not any(c in ids_str for c in comp):
                    match = False
                    break
            # 文字列の場合は「AND条件（必須）」として処理
            else:
                if comp not in ids_str:
                    match = False
                    break
        
        if match:
            results.append((char, ids_str))
            
    return results

# ==========================================
# 3. 使い方：検索したい部品を指定して実行する
# ==========================================

# 例1：「崎」のバリエーションを探す（山、大、可 を含む）
search_target = ["藤"]
results = search_variants(chise_dict, search_target)

print("--- 「崎」のバリエーション検索結果 ---")
for char, ids_str in results:
    print(f"文字/タグ: {char:<15} 分解データ: {ids_str}")
print(f"合計 {len(results)} 件\n")


# 例2：「鳴」のバリエーションを探す（口、鳥 を含む）
search_target2 = ["口", "鳥"]
results2 = search_variants(chise_dict, search_target2)

print("--- 「鳴」のバリエーション検索結果 ---")
for char, ids_str in results2:
    print(f"文字/タグ: {char:<15} 分解データ: {ids_str}")
print(f"合計 {len(results2)} 件")