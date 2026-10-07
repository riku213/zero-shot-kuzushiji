# CodeBookクラス選別とCASIA事前学習サンプルの確認

## 結論

今回の新しい構造CodeBook生成では、`dataset/ids_text/ids.txt` と `dataset/ids_text/ids-cdp.txt` を**直接参照しています**。IDS式の演算子・順序・分解木を取得するのはこの2ファイルです。

一方、`outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl` は、新CodeBookに含めるUnicodeクラス候補の範囲を決めるために参照しました。したがって、「IDSファイルを参照せず、以前のpickleだけから新CodeBookを作った」という意味ではありません。処理は次の二段階です。

1. 以前のpickleから、クラス候補となるUnicode文字を選ぶ。
2. IDSファイルからその文字の分解式を読み、レンダリング済み部首コードを使って新しい構造コードを生成する。

対象は以前のpickle全10,692キーではありません。単一Unicode文字に変換できる4,384キーを候補とし、そのうち43クラスは必要な部品コードを解決できなかったため除外しました。よって、生成済みの新CodeBookに収録されたのは**4,341クラス**です。

## 10,692キーの内訳

`outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl` の10,692キーを、実際のキー文字列で分類した結果です。

| キーの種類 | 件数 | 例 | 今回の扱い |
| --- | ---: | --- | --- |
| `U+XXXX` から単一Unicode文字に変換できるキー | 4,384 | `U+4E00` | クラス候補として選択 |
| 数字だけの複数文字キー | 2,555 | `10`, `100`, `1000` | 1文字のUnicodeクラスではないので対象外 |
| その他の複数文字キー | 3,753 | `CASIA`, `HWDB`, `Train`, `Test`, `asterisk` | 1文字のUnicodeクラスではないので対象外 |
| 合計 | 10,692 |  |  |

最後の2種類、合計6,308キーは6,308種類の漢字ではありません。pickleにはそれらのキーにもベクトル値がありますが、キーが数字ラベルや複数文字の名前なので、6,308種類のUnicode文字に対応する正規クラスとはみなせません。過去のCodeBook生成時に、CASIAのパス・フォルダー名・ファイル名などから文字クラス候補を抽出した結果、数字ラベルや複数文字の名前もキーとして入ったものです。今回のクラス選択処理は、`U+`表記をUnicode文字に変換できるか、またはキー自体がちょうど1文字かを条件にしています。そのため、数字列や `CASIA` のような名前はクラス候補から外れます。

4,384件もすべて漢字とは限りません。かな、記号、ラテン文字などの単一Unicode文字も含まれます。

## 4,384候補から4,341クラスになった理由

4,384は選択対象の候補数であり、生成成功数ではありません。新CodeBook生成時にIDSを解析し、各部品に対応するレンダリング済み64bitコードを探しました。未登録部品はIDS内でさらに分解し、または登録済みの大きな部分木へ置き換えて解決を試みました。それでもコードを作れなかった43クラスは `--unresolved-policy exclude` の指定により除外されました。ランダムコードによる穴埋めはしていません。

除外された43クラス（Unicodeコードポイント）は以下です。

```text
U+0021 U+0023 U+0024 U+0025 U+0028 U+0029 U+002B U+002C U+003D U+0040
U+0047 U+0048 U+0049 U+004A U+004B U+004C U+004D U+004E U+004F U+0051
U+0052 U+0053 U+0054 U+0055 U+0056 U+0057 U+0058 U+0059 U+005A U+005B
U+005D U+005E U+0060 U+007B U+007D U+007E U+3863 U+58F7 U+5E30 U+9B1B
U+FF61 U+FF65 U+29E75
```

実際の生成結果・除外理由は [final_structured_codebook.json](outputs/261005_structured_codebook/final_structured_codebook.json) と [codebook_build.log](outputs/261005_structured_codebook/codebook_build.log) に保存されています。

## CASIAと一致したクラス数

CASIAの画像マニフェストには3,999,566画像がありました。学習スクリプトの既存ラベル解決処理が、画像パスやファイル名から新CodeBookのクラスに対応づけられたのは1,918,886画像・314クラスです。残る2,080,680画像は、この処理ではクラスラベルを解決できませんでした。

事前学習用splitの内訳は以下です。

| split | クラス数 | サンプル数 | 説明 |
| --- | ---: | ---: | --- |
| train | 246 | 1,204,472 | 事前学習に使うクラスとサンプル |
| seen test | 246 | 301,498 | trainと同じクラスだが、trainには入れないサンプル |
| unseen test | 68 | 412,916 | trainクラスに含めないクラスのサンプル |
| 合計 | 314 | 1,918,886 | ラベルを解決できた画像 |

この314は「CASIAに存在する全漢字数」ではなく、「現在のラベル解決ロジックでCodeBookクラスに対応づけられたクラス数」です。マニフェストに画像があっても、CASIA側のディレクトリ名・ファイル名の表記がCodeBookキーへ正しく変換されなければ未解決になります。したがって、残りの画像がCASIAに存在しないとは断定できません。

また、314クラスには `U+0026` (`&`)、`U+003B` (`;`)、`U+0041`〜`U+0046` (`A`〜`F`)、`U+0050` (`P`) も含まれています。これらは一般的な漢字ではありません。名前・パス解析のヒューリスティックが拾った可能性があるため、314クラスすべてを正しいCASIA漢字ラベルとみなす前に、ラベル対応を検証する必要があります。

## CASIAで一致した314クラス

以下は、現行runの `pretrain_training_state.pth` にある314クラスを、新構造CodeBookのキーと照合した一覧です。文字環境による表示差を避けるため、Unicodeコードポイントで列挙します。各コードポイントは対応する1文字を表します。

```text
U+0026 U+003B U+0041 U+0042 U+0043 U+0044 U+0045 U+0046 U+0050 U+6248 U+6FF6 U+7AB6 U+7ACA U+7ACF U+7AD3 U+7AD5
U+7B18 U+7B33 U+7DD5 U+7E32 U+7E7D U+8373 U+8375 U+8389 U+83A0 U+83A8 U+83EB U+83F4 U+8413 U+86DB U+86DF U+86EF
U+86F9 U+86FB U+8700 U+8703 U+8706 U+8708 U+8709 U+870A U+870D U+8711 U+8712 U+871A U+8725 U+8729 U+8734 U+8737
U+873B U+873F U+874C U+874E U+8753 U+8757 U+8759 U+875F U+8760 U+8763 U+8768 U+876A U+876E U+8774 U+8778 U+8782
U+879F U+87A2 U+87AB U+87AF U+87B3 U+87BB U+87BD U+87C0 U+87C4 U+87C6 U+87C7 U+87CB U+87D0 U+87D2 U+87E0 U+87EF
U+87F2 U+87F6 U+87F7 U+87FE U+8805 U+880D U+880E U+880F U+8811 U+8815 U+8816 U+8822 U+894D U+8ADB U+8ADE U+8AE0
U+8AE1 U+8AE2 U+8AE4 U+8AF1 U+8AF7 U+8B07 U+8B0C U+8B10 U+8B14 U+8B16 U+8B17 U+8B1A U+8B20 U+8B26 U+8B28 U+8B2B
U+8B33 U+8B3E U+8B41 U+8B49 U+8B4C U+8B4E U+8B4F U+8B56 U+8B5A U+8B5B U+8B5F U+8B6B U+8B6C U+8B6F U+8B74 U+8B7D
U+8B80 U+8B8C U+8B8E U+8B92 U+8B93 U+8B96 U+8B99 U+8C3F U+8C41 U+8C48 U+8C4C U+8C4E U+8C50 U+8C55 U+8C62 U+8C6C
U+8C78 U+8C7A U+8C7C U+8C82 U+8C85 U+8C89 U+8C8A U+8C8D U+8C8E U+8C94 U+8F62 U+8F63 U+8F64 U+8F9C U+8F9F U+8FA3
U+8FAD U+8FAF U+8FB7 U+8FDA U+8FE2 U+8FE5 U+8FEA U+8FEF U+8FF4 U+8FF8 U+8FF9 U+8FFA U+9005 U+900B U+900D U+900E
U+9011 U+9015 U+9016 U+901E U+9021 U+9027 U+9035 U+9036 U+9039 U+903E U+9049 U+904F U+9050 U+9051 U+9052 U+9056
U+9058 U+905E U+9068 U+906F U+9072 U+9076 U+907D U+9080 U+9081 U+9082 U+9083 U+9087 U+9089 U+908A U+908F U+90DB
U+90E2 U+90E4 U+9102 U+9112 U+9119 U+95A0 U+95A7 U+95A8 U+95AD U+95B9 U+95BB U+95BC U+95BE U+95C3 U+95CA U+95CC
U+95CD U+95D4 U+95D5 U+95D6 U+95DC U+95E1 U+95E2 U+95E5 U+9621 U+9628 U+962E U+962F U+9642 U+964B U+964C U+964F
U+965C U+965D U+965E U+965F U+9666 U+966C U+9672 U+9677 U+968D U+9695 U+9697 U+9698 U+96A7 U+96A8 U+96AA U+96B0
U+96B1 U+96B8 U+96B9 U+96CB U+96CD U+96CE U+96D5 U+96D6 U+96DC U+96F9 U+9704 U+9706 U+970D U+970E U+9711 U+97AB
U+9A3E U+9A4D U+9A55 U+9A57 U+9A5B U+9A5F U+9A62 U+9A65 U+9A6A U+9A6B U+9ABC U+9AC0 U+9AD3 U+9AD4 U+9ADE U+9ADF
U+9AE6 U+9AEB U+9AEE U+9AEF U+9AF1 U+9AF4 U+9AF7 U+9AFB U+9B1A U+9B1F U+9B22 U+9B23 U+9B25 U+9B2A U+9B2E U+9B2F
U+9B32 U+9B4E U+9B51 U+9B91 U+9B96 U+9B97 U+9B9F U+9BA0 U+9BA8 U+9BB4
```

## CASIAに未マッチのクラス一覧

新構造CodeBookに収録された4,341クラスのうち、現在のCASIA事前学習splitに一度も現れないクラスは**4,027件**です。4,384候補全体を基準にすれば、CASIAにマッチしないものは4,070件で、内訳は「新CodeBookに収録されたが未マッチ」4,027件と「CodeBook生成時に除外」43件です。

4,027件をこの文書に重複転記すると、CodeBook本体やsplitが更新された際に一覧が古くなるおそれがあります。次のコマンドは、実際に使ったpickleとcheckpointから**全4,027件を文字とU+コードポイントで列挙**します。出力がこのデータセットでの未マッチ一覧です。

```python
import pickle
import torch
from pathlib import Path

root = Path(".")
with (root / "outputs/261005_structured_codebook/final_structured_codebook.pkl").open("rb") as f:
	codebook = pickle.load(f)
state = torch.load(
	root / "outputs/261005_structured_codebook/training/pretrain_training_state.pth",
	map_location="cpu",
	weights_only=False,
)
matched = set(state["class_to_index"])
unmatched = sorted(set(codebook["codes"]) - matched, key=lambda key: int(key[2:], 16))
for key in unmatched:
	char = chr(int(key[2:], 16))
	print(f"{char}\t{key}")
print(f"unmatched={len(unmatched)}")
```

この未マッチ集合は、CASIA画像が存在しないと確認された文字一覧ではありません。あくまで、今回のmanifest・ファイル名/フォルダー名ラベル解決・許可クラス集合を使ったsplitでサンプルが0件だったクラスです。

## 再現用の参照ファイル

- 元クラス候補: `outputs/260901_codebook_CASIA/final_codebook_with_casia.pkl`（10,692キー）
- 新構造CodeBook: `outputs/261005_structured_codebook/final_structured_codebook.pkl` と同名JSON
- CASIAの実split情報: `outputs/261005_structured_codebook/training/pretrain_training_state.pth`
- train / seen test / unseen test の画像パス一覧: 同じtrainingディレクトリの3つの `pretrain_*_manifest.txt`
- 実行ログと未解決診断: `outputs/261005_structured_codebook/training/train.log` と `outputs/261005_structured_codebook/codebook_build.log`
