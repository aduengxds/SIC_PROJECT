# Detecting and counting flooded houses from aerial imagery

[Tiếng Việt](README.md) · **English** · [中文](README.zh.md) · [Français](README.fr.md) · [한국어](README.ko.md) · [日本語](README.ja.md)

## Team

| Member | GitHub |
| --- | --- |
| Phùng Duy Hoàng | [@21xDHoang](https://github.com/21xDHoang) |
| Bùi Nam Khánh | [@NK-Lync](https://github.com/NK-Lync) |
| Tào Hoàng Minh Vũ | [@aduengxds](https://github.com/aduengxds) |

A two-stage pipeline: a **detector** locates every house in an aerial image, a **classifier** reads the state of the cropped region, and the houses are then counted per class.

- Detector: **Cascade R-CNN + ResNet-50-FPN**, a single `house` class
- Classifier: **ConvNeXt-Tiny**, three classes `Flooded` / `Not-Flooded` / `Partially-Flooded`
- Web app: **FastAPI**, running entirely locally on a machine with a GPU

| Item | Metric | Value |
| --- | --- | --- |
| Detector (`best_ep4`) | AP50 | 0.631 |
| Classifier | end-to-end accuracy | 0.6682 |
| Counting @ threshold 0.30 | 4,287 boxes / 3,441 labelled boxes | +24.6% |

> The default counting threshold of 0.30 favours recall, so the total is **24.6% over-counted**, while per-image error stays large (MAE 10.95; p90 18; max 49). The pipeline suits a rough survey and is **not accurate enough to count a single image** — see *Known limitations*.

## The problem

| | |
| --- | --- |
| Input | one 640×640 aerial image |
| Output | a box per house, each carrying one of three states, plus a per-class count table |

Main difficulties: houses are very small, the water is murky, the `Flooded` / `Partially-Flooded` labels are ambiguous, and an image may be empty or extremely crowded.

## Data

Source: Roboflow, 640×640 images, three classes `Flooded`, `Not-Flooded`, `Partially-Flooded`. Total **2,778 images · 44,967 boxes**; the split is taken as-is from Roboflow, and the test set is used only once, for reporting.

| Attribute | Value |
| --- | --- |
| Dataset | [`merged_dataset` – v1](https://universe.roboflow.com/housedetectionclassification-jugwz/merged_dataset-dk0ch) |
| License | **CC BY 4.0** — attribution is required on reuse |
| Label format | COCO |
| Roboflow preprocessing | Auto-orient (EXIF stripped), resize 640×640 stretch; no augmentation |
| Export date | 7 Nov 2024 |

| Split | Images | Boxes | Notes |
| --- | --- | --- | --- |
| Train | 2,116 | 33,145 | boxes < 32 px: 10.7%; < 64 px: 48.5% |
| Valid | 459 | 8,381 | used to select the model and the threshold |
| Test | 203 | 3,441 | 12 images contain no houses |

Full statistics: `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt`.

## Architecture

The two models learn different jobs: the detector only answers "where is a house", the classifier only "what state is it in". At inference, crops come from detector boxes — ground-truth labels are not used.

**Detector** — Cascade R-CNN, ResNet-50-FPN backbone, 3 stages with IoU 0.5 / 0.6 / 0.7, one `house` class + background (2 logits), anchors 32–512 px (stride 4–64 × scale 8), `keep_ratio` resize pipeline at 1333×800, no `RandomResize`. Config: `huan_luyen/cau_hinh_detector.py`.

**Classifier** — ConvNeXt-Tiny (ImageNet-1K V1), 27.82 M parameters, 768 → 3 output, 224 px input, 15% crop margin at both training and inference, 12 epochs of AdamW at 3e-4, label smoothing 0.05.

Default pipeline: image → detector → keep score ≥ 0.30, at most 100 boxes → crop (+15%) → ConvNeXt-Tiny → count per class.

## Retraining

Two notebooks run on Colab (a T4 is enough); open one and hit *Run all* top to bottom:

- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002_Aug.ipynb` — the augmented version, which produces the `best_ep4` in use
- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002.ipynb` — the non-augmented version; its epoch 20 numbers are in `02_ket_qua_detector.txt`

| Cells | What they do |
| --- | --- |
| 1–3 | environment check, install `mmcv` / `mmengine` / `mmdet`, verify that CUDA ops actually run |
| 4–5 | fetch the data, compute statistics, convert to two formats: a single `house` class for the detector and the original three classes for the classifier |
| 6–7 | train the Cascade R-CNN detector, plot learning curves, evaluate mAP on test |
| 8 | train the ConvNeXt-Tiny classifier on crops taken from real boxes |
| 9–10 | run detect → classify → count over 203 images, summarise and write the 4 result files |
| 11 | Gradio web demo |

Checkpoints are saved by `save_best` on validation mAP, so `best_ep4` is **epoch 4**, not the last epoch — do not confuse it with `epoch_20` from the non-augmented run.

## Results

![Detection results on an aerial image — red boxes are flooded, orange partially flooded, green not flooded](export/mau_ket_qua.jpg)

*Image exported by the web app itself (`export/mau_ket_qua.jpg`). Box colours follow `loi_suy_luan.py`: red = `Flooded`, orange = `Partially-Flooded`, green = `Not-Flooded`.*

### Detector

Results exist for two checkpoints, **do not mix them**:

| Checkpoint | AP | AP50 | AP75 | AP_s | Source |
| --- | --- | --- | --- | --- | --- |
| `best_ep4` (web app default) | 0.313 | 0.631 | 0.284 | 0.169 | figures presented in the report |
| epoch_20 (no-aug) | 0.277 | 0.584 | 0.233 | 0.149 | `02_ket_qua_detector.txt` |

The low AP_s is a real limitation: the smallest RPN anchor is 32 px (stride 4 × scale 8) and training uses a single resize pipeline, so houses only a few tens of pixels across have no well-matched anchor.

### Classifier

Over 8,381 valid crops: **accuracy 0.7195**. On test crops (ground truth): 0.6713. Using detector boxes: **end-to-end 0.6682** — barely lower, so the detector's crop is not the bottleneck.

| True \ Predicted | Flooded | Not-Flooded | Partially-Flooded | Recall |
| --- | --- | --- | --- | --- |
| Flooded | 1,514 | 115 | 606 | 67.7% |
| Not-Flooded | 55 | 2,696 | 608 | 80.3% |
| Partially-Flooded | 440 | 527 | 1,820 | 65.3% |

Errors concentrate in the `Flooded` ↔ `Partially-Flooded` pair (606 + 440). The classifier has **no background class**, so junk boxes from the detector are still forced into one of the three classes.

### Counting

| Checkpoint | Boxes @ 0.30 | Labels | Difference | MAE / image with houses |
| --- | --- | --- | --- | --- |
| `best_ep4` | 4,287 | 3,441 | +846 (+24.6%) | 10.95 (p90 18, max 49) |
| epoch_20 | 3,414 | 3,441 | −27 (−0.8%) | 9.47 |

epoch_20's total nearly matches the labels, but that is **error cancellation**: some images over-count while others under-count, so per-image MAE is still near 10. Do not read the total in isolation.

## Metric definitions

| Metric | Definition / scope |
| --- | --- |
| AP | COCO bbox, averaged over IoU 0.50 : 0.05 : 0.95 |
| AP50 / AP75 | IoU 0.50 / 0.75 |
| AP_s / AP_m / AP_l | boxes with area < 32² / 32²–96² / > 96² px² |
| Accuracy (classifier) | correct / total, measured on the 8,381 valid crops |
| End-to-end | classifier run on detector boxes matched to labels at IoU 0.5 |
| MAE | mean \|predicted count − label count\| over images that contain houses |

AP is measured at `score_thr` 0.05 while counting uses a 0.30 threshold — different scales, so do not compare AP with accuracy directly.

## Verification

Re-running all 203 images with the web app's own code (`best_ep4`, threshold 0.30) gives 4,288 boxes, off by **+1 box over 4,287 = 0.023%**. The `Flooded` class matches exactly at 890/890; the gap comes from OpenCV and GPU architecture differences between Colab (Tesla T4) and the web machine (RTX 2050). The 4,287 target is hard-coded as `MOC_4287` in `kiem_tra_moi_truong.py` so the check can be reproduced.

Details: `nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt`.

## Documents in this repo

| File | Contents |
| --- | --- |
| [`HUONG_DAN_CHO_BAN.txt`](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.txt) | Complete installation and running guide, with a table of common errors |
| [`KET_LUAN_KIEM_CHUNG.txt`](nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt) | Record of the 4,287-box test and an explanation of the +1 difference |
| [`01_thong_ke_du_lieu.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt) | Statistics for 2,778 images · 44,967 boxes per split |
| [`02_ket_qua_detector.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/02_ket_qua_detector.txt) | Per-metric AP for the epoch_20 checkpoint |
| [`03_ket_qua_classifier.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/03_ket_qua_classifier.txt) | Confusion matrix and accuracy of the classifier |
| [`04_dem_nha_test.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/04_dem_nha_test.txt) | Per-class count table for the epoch_20 checkpoint |

## Repository layout

```
.
├── export/                              # files exported by the app: png/, jpg/, csv/
├── nha_ngap_v002_code/nha_ngap_v002_code/
│   ├── huan_luyen/                      # Colab notebooks, detector config, 4 result files
│   └── web/                             # older web copy, kept for comparison
└── nha_ngap_web/nha_ngap_web/           # the main application
    ├── ung_dung.py                      # FastAPI: UI and model lifecycle
    ├── loi_suy_luan.py                  # pure inference, knows nothing about the web
    ├── kiem_tra_moi_truong.py           # 5-step environment check, has a --dem-test flag
    ├── giao_dien/index.html
    ├── checkpoints/                     # cau_hinh_detector.py (present); 2 .pth files (add yourself)
    └── test/                            # 203 test images
```

## Installation and running

### Requirements

| Item | Value |
| --- | --- |
| OS | Windows 10/11 64-bit |
| Python | **3.13 64-bit** from python.org — not the Microsoft Store build; 3.12 and 3.14 cannot install `mmcv` |
| GPU | NVIDIA; verified on an RTX 2050 4 GB |
| Driver | R580 or newer (CUDA 13.2) |
| Disk | ~6–7 GB; first install downloads ~2.5 GB |
| Path | ASCII, no accents, no spaces |

### Checkpoints

The repo does **not** include `.pth` files (`.gitignore` excludes `**/*.pth`). All three files are required under `nha_ngap_web/nha_ngap_web/checkpoints/`:

```
work_det/cau_hinh_detector.py                     already in the repo
work_det/work_aug/best_coco_bbox_mAP_epoch_4.pth  add yourself
work_cls/convnext_tiny_3lop.pth                   add yourself
```

Missing any one of the three and the web app will not start. The two `.pth` files come from the training machine or from the team's Drive (`nha_ngap_v002_out/`).

### Steps

```bat
cd nha_ngap_web\nha_ngap_web

:: create the venv and download ~2.5 GB, takes 10-30 minutes
CAI_DAT.bat

:: must pass all 5 steps
venv\Scripts\python.exe kiem_tra_moi_truong.py

:: opens http://127.0.0.1:8000
CHAY_WEB.bat
```

If port 8000 is taken, use `CHAY_WEB.bat --cong 8080`.

To reproduce the counting figure in the report (2–3 minutes, should give about 4,287 boxes):

```bat
venv\Scripts\python.exe kiem_tra_moi_truong.py --dem-test
```

The web app has four endpoints: `GET /`, `GET /api/thong_tin`, `POST /api/du_doan`, `POST /api/export` (saves PNG/JPG/CSV into `export/`). The model is loaded once at start-up; because `mmdet` is not thread-safe and the GPU has only 4 GB, requests are queued rather than run in parallel.

## Known limitations

- **Small, single-source test set** — 203 images, no external test.
- **Ambiguous labels** — `Flooded` and `Partially-Flooded` account for most classifier error.
- **Split still shares sources** — 7 MD5 groups, 49 base names, 3 video groups; deduplication and source-aware splitting are needed.
- **A correct total does not mean correct per-image counts** — MAE 10.95; p90 18; max 49.
- **No background class** — junk boxes are forced into one of the three classes, inflating counts.
- **Threshold not optimised** — scores are not calibrated; raising the threshold only trades precision for recall, it does not fix the cause.
- **Small houses** — low AP_s due to the smallest anchor and single-scale training.
- **Not production-ready** — a background class, calibration and a clean split are needed before real counting.

## Improvement roadmap

| Priority | Work | Why |
| --- | --- | --- |
| P0 | Add a background / reject class to the classifier | junk boxes are currently labelled as houses — the largest source of over-counting |
| P0 | Calibrate scores on the validation set | a 0.30 threshold currently carries no probabilistic meaning |
| P1 | Choose the threshold by count MAE rather than F1 | the goal is counting, not detection |
| P1 | Anchors / tiling / random scale for small houses | AP_s reaches only 0.169 |
| P2 | Deduplicate and split by source | a cleaner test set makes the numbers fully trustworthy |
| P2 | Raise `max_per_img` to 150–200 | 2 of 203 images are still capped at 100 boxes |

## License

- **Source code**: MIT — see [`LICENSE`](LICENSE).
- **Data**: Roboflow `merged_dataset` v1, licensed CC BY 4.0 — attribution is required on reuse.

---

*Translated from [README.md](README.md). If any figure differs, the Vietnamese version is authoritative.*
