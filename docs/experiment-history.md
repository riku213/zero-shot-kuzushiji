# Experiment History and Codebook Provenance

This document tracks experiment intent and artifact relationships. Generated files remain under the ignored `outputs/` directory; this document is the compact, version-controlled index to them. Dates below use `YYMMDD` where that is how the output directories are named.

## Codebook Pipeline

The legacy pipeline in `src/3_extract_features.py` extracts ResNet-34 features from rendered glyphs, and `src/4_generate_fare.py` projects those features to 64 dimensions and binarizes them. `radical_codes.pkl`/`.json` is the glyph-to-64-bit-code mapping. Structured class codebooks are built separately by `src/8_build_structured_codebook.py` by combining 4-bit IDS structure symbols with terminal glyph codes.

`src/3_extract_features.py` defaults to ImageNet-pretrained ResNet-34. Fine-tuned weights are used only when passed through `--weights`. A feature artifact records its source manifest but does not record the weights path, so the exact backbone used for an existing feature file cannot always be established from that file alone.

## Verified Codebook Runs

| Output directory | Role / generation | Notes |
| --- | --- | --- |
| `outputs/` | Initial FaRE artifacts, 2026-08-22 | `radical_codes.pkl` and `.json`; 9,730 entries. This is not the latest dictionary. Its source is `outputs/radical_features.pkl`. |
| `outputs/261005_structured_codebook/` | Rebuilt 64-bit glyph codes and structured codebook, 2026-10-05 | `radical_codes.pkl` has 9,730 entries. The final structured codebook records this input path. |
| `outputs/261007_correct_dict/` | Combined Kuzushiji/CASIA glyph inventory and codes, 2026-10-07 | `all_glyph_codes.pkl`; 10,016 entries. Adds glyphs beyond the earlier radical inventory. |
| `outputs/261009_resnet_params/` | ResNet-34 SupCon fine-tuning, 2026-10-09/10 | Contains `resnet34_supcon_best.pth` and `resnet34_supcon_last.pth`. |
| `outputs/261010_tuned_codebook/` | Feature/codebook run, 2026-10-10 | `all_glyph_features.pkl` differs for all 10,016 records from the 2026-10-07 features. The artifact records its manifest but not the backbone weights path; available command history does not prove which weights were passed to feature extraction. Its `all_glyph_codes.pkl` mapping is exactly equal to the 2026-10-07 and 2026-10-18 ordinary-code mappings. |
| `outputs/261018_cos_codebook/` | Cosine-metric UMAP codes, 2026-10-18 | `all_glyph_codes.pkl`; its 10,016 code vectors exactly match the 2026-10-07 and 2026-10-10 ordinary mappings in the checked artifacts. |
| `outputs/261018_cos_median_codebook/` | Median-threshold code dictionary, 2026-10-18 | `all_glyph_codes.pkl`; 10,016 entries. Its code mapping differs from the ordinary-threshold mapping. This is the 64-bit glyph dictionary referenced by the later 400-dimensional structured codebook. |
| `outputs/261019_ViT_umap_cos/` | ViT run using cosine/median glyph codes | `final_codebook_1212.json` identifies `outputs/261018_cos_median_codebook/all_glyph_codes.pkl` as its source dictionary. |
| `outputs/261024_ViT_code400/` | ViT with 400-dimensional structured targets, 2026-10-24/25 | `final_codebook_400.json` identifies `outputs/261018_cos_median_codebook/all_glyph_codes.pkl` as the source glyph dictionary. November 2026 smoke/stage experiments also reference this final codebook. |

The equality statements above were checked by comparing the complete `radical_to_code` mappings, not inferred from filenames or timestamps. File timestamps indicate artifact chronology but do not alone establish which run is authoritative.

## Thresholding and Duplicate-Code History

- The early 64-dimensional UMAP pipeline binarized around zero. In a run where projected values were all positive, every vector became all `+1`, causing duplicate codes.
- When a projected binary vector collided with an existing vector, `src/4_generate_fare.py` generated a replacement from a seeded RNG. This is reproducible for fixed inputs, ordering, and seed, but later replacement codes depend on which collisions occurred earlier in that order; it is not a per-input hash mapping.
- Repeated 2026-10-07, 2026-10-10, and 2026-10-18 ordinary dictionaries were verified identical at the full mapping level. This is consistent with deterministic regeneration and collision handling, though it does not by itself prove every intermediate feature vector or run command was identical.
- Median thresholding was introduced to avoid the all-positive/zero-threshold collapse. The 2026-10-18 median-threshold dictionary has different code values and is the source recorded by the later structured codebooks.

## 2026-11 Tree-Output Experiments

The 2026-11-02 experiments use the same `outputs/261024_ViT_code400/final_codebook_400.pkl` target codebook, with initialization from `outputs/261027_ViT_code400_v2/training/pretrain_best_fare_model.pth`.

| Output directory | Experiment intent |
| --- | --- |
| `outputs/261102_priority01_smoke/` | Smoke check for the updated unseen-class evaluation and split workflow. |
| `outputs/261102_priority2_smoke/` | Follow-up smoke check for the second priority. |
| `outputs/261102_stage1_ft/` | Fine-tuning stage with unseen-class validation/test and ZSL/GZSL evaluation updates. |
| `outputs/261102_stage2_ft/` | Experiment with output heads for tree nodes, training each component as a classification task. |

The exact run arguments and metrics are recorded in `command.md` and the respective output metadata/logs. Add a short row here when a new experiment directory is created; include its purpose and the codebook/checkpoint inputs so later work does not have to infer provenance from directory names.

## Experiment Results (Japanese)

個別実験の日本語レポートは `docs/experiment-results/` に格納する。

| 実験 | レポート | 主な結果 |
| --- | --- | --- |
| 2026-11-02 Stage 2 木構造部品分類 | [261102-stage2-ft.md](experiment-results/261102-stage2-ft.md) | 最良epoch 1 (`val_h=0.4300`)、test seen GZSL 71.23%、unseen GZSL 22.09%、component融合 unseen ZSL 35.69%。 |
| 2026-11-19 固定592-bit ViT | [261119-fixed-bit-vit-results.md](261119-fixed-bit-vit-results.md) | CASIA事前学習後にくずし字を微調整。test seen 77.69%、unseen 16.56% (GZSL)。 |
| 2026-11-25 Stage 1分類 + Stage 2 oracle基準実験 | [261125-two-stage-m3-baseline.md](experiment-results/261125-two-stage-m3-baseline.md) | 256クラス、2,427画像、各5 epoch。Stage 1 test skeleton 74.26%、Stage 2 oracle slot 17.73%、文字精度4.73%。 |

## Proposed Fixed 592-Bit Tree Format

The proposed layout is one 4-bit root structure slot, three 4-bit child-structure slots, and nine fixed 64-bit terminal slots: `4 + 3 * (4 + 3 * 64) = 592`. Operator nodes may occur only at tree depths 0 and 1; depth 2 is terminal codes only. `SINGLE` marks an unsplit glyph and routes its 64-bit code to the first leaf slot; unused child and leaf slots are zero.

An earlier audit counted only total operator/leaf capacity and incorrectly concluded that all 4,328 Kuzushiji classes fit. Rechecking `outputs/261018_cos_median_codebook/all_glyph_codes.pkl` with the depth-position constraint and existing component decomposition/replacement logic found:

- 4,287 / 4,328 classes fit while preserving the IDS root structure and encoding each child subtree within one branch block.
- 30 additional classes have a direct 64-bit class/glyph code and can fit only if the whole class is deliberately collapsed to `SINGLE`, losing its internal IDS structure in the target representation.
- 11 classes do not fit with the checked dictionary and current decomposition rules: 5 have unresolved CDP terminal components; 6 require a deeper branch that has no usable registered glyph code for the needed subtree.

These are structural representability counts, not model accuracy results. The audit is reproducible with `src/inspect_ids_depth.py`; it defaults to the median-threshold glyph dictionary above. The exact dictionary and component-resolution policy must be recorded when rerunning it.

The 4-bit symbol assignment still needs to be defined for this format. `NONE=0000` requires every IDS operator and `SINGLE` to use nonzero values. The existing `STRUCTURE_BITS` enumeration assigns the first IDS operator index zero, so it cannot be copied as-is if its bit representation uses `0000` for that operator. There are enough remaining 4-bit values for 12 IDS operators plus `SINGLE`, but the mapping must be explicit and tested.

## 2026-11-19 Fixed 592-Bit ViT

The requested symbol mapping is now fixed: `NONE=0000`, `SINGLE=0001`, then the 12 operators in existing `STRUCTURE_SYMBOLS` order assigned `0010` through `1101`; `1110` and `1111` are reserved. Bit zero/one maps to -1/+1 in the FaRE target. All 16 structure-symbol bits are valid targets, while absent leaf slots are zero and masked out. Deep subtrees are collapsed only when an exact registered glyph code exists. The 41 Kuzushiji classes failing strict layout are excluded, including classes that would otherwise fit by discarding their IDS tree as whole-class `SINGLE`.

New files implement the separate pipeline without changing the existing codebook or training scripts:

- `src/11_build_fixed_592_codebook.py` builds the fixed-layout codebook from the 2026-10-18 median glyph codes and 2026-10-07 combined render metadata/class-union manifest.
- `src/12_train_fixed_592_vit.py` invokes the existing ViT-B/16 training/evaluation workflow, verifies the fixed 592-bit codebook, starts from ImageNet weights, and defaults to CASIA pretraining followed by Kuzushiji training.
- `tests/test_fixed_592_codebook.py` tests symbol assignment, binary/nested layouts, exact-subtree replacement, `SINGLE`, and missing-code rejection.
- `docs/fixed-592-bit-training.md` records the specification and run commands in Japanese.

The generated ignored artifact `outputs/261119_fixed_bit_ViT/fixed_592_codebook.pkl` contains 9,159 classes from the 9,213-class union: 4,287 Kuzushiji classes and 4,872 CASIA-only classes. The 41 non-fitting Kuzushiji classes and 13 additionally unencodable CASIA-only classes are listed in `codebook_exclusions.json`. Codebook-missing classes are filtered by the existing data collectors from pretraining, training, validation, and test splits.

学習結果の日本語レポート: [docs/261119-fixed-bit-vit-results.md](261119-fixed-bit-vit-results.md)。CASIA事前学習とくずし字ファインチューニングのテスト指標、選択checkpoint、設定、損失曲線を記録した。出力側のレポートとグラフは `outputs/261119_fixed_bit_ViT/training/result/` にあり、再作図スクリプトは `outputs/261119_fixed_bit_ViT/training/plot_loss_curves.py`。