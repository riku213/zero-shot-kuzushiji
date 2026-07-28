import urllib.request
import re

url = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"

def load_ids_data(url):
    ids_dict = {}
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        for line in response.read().decode('utf-8').splitlines():
            if line.startswith(';'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ids_dict[parts[1]] = parts[2]
    return ids_dict

chise_dict = load_ids_data(url)
ids_operators = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# 【追加】これ以上分解しない「部首・基本部品」のリスト（必要に応じて追加）
stop_radicals = set("艹月氺水日木人手心") 

def decompose_practical(char, ids_dict, max_depth=10, current_depth=0):
    if current_depth > max_depth or char in ids_operators:
        return char
    
    if char in stop_radicals:
        return char

    if char not in ids_dict or ids_dict[char] == char:
        return char
        
    first_candidate = ids_dict[char].split(',')[0]
    
    first_candidate = re.sub(r'\[.*?\]', '', first_candidate)
    
    full_decomposition = ""
    for sub_char in first_candidate:
        full_decomposition += decompose_practical(sub_char, ids_dict, max_depth, current_depth + 1)
        
    return full_decomposition

# テスト実行
target_kanji = "藤"
result = decompose_practical(target_kanji, chise_dict)
print(f"【{target_kanji}】: {result}")