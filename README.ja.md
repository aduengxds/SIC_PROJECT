# 空撮画像からの浸水家屋の検出と計数

[Tiếng Việt](README.md) · [English](README.en.md) · [中文](README.zh.md) · [Français](README.fr.md) · [한국어](README.ko.md) · **日本語**

## チーム

| メンバー | GitHub |
| --- | --- |
| Phòng Duy Hoàng | [@21xDHoang](https://github.com/21xDHoang) |
| Bùi Nam Khánh | [@NK-Lync](https://github.com/NK-Lync) |
| Tào Hoàng Minh Vũ | [@aduengxds](https://github.com/aduengxds) |

2 段階のパイプラインです。**検出器**が空撮画像内の各家屋の位置を特定し、**分類器**が切り出した領域の状態を判定し、最後にクラスごとに家屋数を数えます。

- 検出器: **Cascade R-CNN + ResNet-50-FPN**、単一の `house` クラス
- 分類器: **ConvNeXt-Tiny**、`Flooded` / `Not-Flooded` / `Partially-Flooded` の 3 クラス
- Web アプリ: **FastAPI**、GPU を搭載したマシン上で完全にローカル動作

| 項目 | 指標 | 値 |
| --- | --- | --- |
| 検出器（`best_ep4`） | AP50 | 0.631 |
| 分類器 | エンドツーエンド精度 | 0.6682 |
| 計数（しきい値 0.30） | 4,287 ボックス / 3,441 正解ボックス | +24.6% |

> 既定の計数しきい値 0.30 は再現率を優先するため、総数は **24.6% 過大**になります。一方で 1 枚あたりの誤差は依然として大きく（MAE 10.95、p90 18、最大 49）、このパイプラインは概況把握には適しますが **1 枚の画像を正確に数えるには不十分です** — *既知の制約* を参照してください。

## 課題

| | |
| --- | --- |
| 入力 | 640×640 の空撮画像 1 枚 |
| 出力 | 家屋ごとのボックス（3 状態のいずれか）と、クラス別の計数表 |

主な難点は、家屋が非常に小さいこと、水が濁っていること、`Flooded` と `Partially-Flooded` のラベル境界が曖昧なこと、画像が完全に空の場合や極端に密集している場合があることです。

## データ

出典 Roboflow、640×640 画像、3 クラス `Flooded`、`Not-Flooded`、`Partially-Flooded`。合計 **2,778 枚 · 44,967 ボックス**。分割は Roboflow から受け取ったままを使用し、テストセットは報告時に一度だけ使います。

| 属性 | 値 |
| --- | --- |
| データセット | [`merged_dataset` – v1](https://universe.roboflow.com/housedetectionclassification-jugwz/merged_dataset-dk0ch) |
| ライセンス | **CC BY 4.0** — 再利用時は出典表示が必須 |
| ラベル形式 | COCO |
| Roboflow の前処理 | 自動向き補正（EXIF 削除）、640×640 への引き伸ばしリサイズ、データ拡張なし |
| エクスポート日 | 2024-11-07 |

| 分割 | 画像 | ボックス | 備考 |
| --- | --- | --- | --- |
| Train | 2,116 | 33,145 | 32 px 未満のボックスが 10.7%、64 px 未満が 48.5% |
| Valid | 459 | 8,381 | モデルとしきい値の選択に使用 |
| Test | 203 | 3,441 | うち 12 枚は家屋を含まない |

詳細な統計: `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt`.

## アーキテクチャ

2 つのモデルは別の仕事を学びます。検出器は「家屋がどこにあるか」だけを、分類器は「どんな状態か」だけを答えます。推論時のクロップは検出器のボックスから得ており、正解ラベルは使いません。

**検出器** — Cascade R-CNN、ResNet-50-FPN バックボーン、IoU 0.5 / 0.6 / 0.7 の 3 ステージ、`house` クラス + 背景（2 logit）、アンカー 32–512 px（stride 4–64 × scale 8）、`keep_ratio` リサイズパイプライン 1333×800、`RandomResize` なし。設定: `huan_luyen/cau_hinh_detector.py`.

**分類器** — ConvNeXt-Tiny（ImageNet-1K V1）、27.82 M パラメータ、出力 768 → 3 クラス、入力 224 px、学習と推論の両方で 15% のクロップ余白、12 エポック、AdamW 3e-4、label smoothing 0.05。

既定のパイプライン: 画像 → 検出器 → score ≥ 0.30、最大 100 ボックス → クロップ（+15%）→ ConvNeXt-Tiny → クラス別に計数。

## 再学習

2 つのノートブックは Colab で動作します（T4 で十分）。開いて上から *Run all* を実行してください:

- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002_Aug.ipynb` — 拡張ありの版。現在使用している `best_ep4` を生成
- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002.ipynb` — 拡張なしの版。エポック 20 の数値は `02_ket_qua_detector.txt` にあり

| セル | 内容 |
| --- | --- |
| 1–3 | 環境確認、`mmcv` / `mmengine` / `mmdet` のインストール、CUDA 演算が実際に動くかの確認 |
| 4–5 | データ取得、統計、2 形式への変換: 検出器用の単一 `house` クラスと、分類器用の元の 3 クラス |
| 6–7 | Cascade R-CNN 検出器の学習、学習曲線の描画、テストでの mAP 評価 |
| 8 | 実際のボックスからクロップしたサンプルで ConvNeXt-Tiny 分類器を学習 |
| 9–10 | 203 枚で 検出 → 分類 → 計数 を実行し、集計して結果ファイル 4 つを書き出す |
| 11 | Gradio の Web デモ |

チェックポイントは検証 mAP による `save_best` で保存されるため、`best_ep4` は **4 エポック目**であり最終エポックではありません。拡張なし版の `epoch_20` と混同しないでください。

## 結果

![空撮画像での検出結果 — 赤は浸水、橙は一部浸水、緑は非浸水](export/mau_ket_qua.jpg)

*この画像は Web アプリ自身が出力したものです（`export/mau_ket_qua.jpg`）。ボックスの色は `loi_suy_luan.py` に従います: 赤 = `Flooded`、橙 = `Partially-Flooded`、緑 = `Not-Flooded`。*

### 検出器

2 つのチェックポイントの結果があります。**数値を混ぜないでください**:

| チェックポイント | AP | AP50 | AP75 | AP_s | 出典 |
| --- | --- | --- | --- | --- | --- |
| `best_ep4`（Web アプリの既定） | 0.313 | 0.631 | 0.284 | 0.169 | 報告書に示した数値 |
| epoch_20（拡張なし） | 0.277 | 0.584 | 0.233 | 0.149 | `02_ket_qua_detector.txt` |

AP_s の低さは実際の制約です。RPN の最小アンカーが 32 px（stride 4 × scale 8）で、学習も単一のリサイズパイプラインのみを使うため、数十ピクセル程度の家屋には適したアンカーが存在しません。

### 分類器

検証セットの 8,381 クロップで **精度 0.7195**。テストのクロップ（正解ラベル）では 0.6713。検出器のボックスを使うと **エンドツーエンドで 0.6682** となり、ほとんど低下しません。つまり検出器のクロップはボトルネックではありません。

| 真値 \ 予測 | Flooded | Not-Flooded | Partially-Flooded | 再現率 |
| --- | --- | --- | --- | --- |
| Flooded | 1,514 | 115 | 606 | 67.7% |
| Not-Flooded | 55 | 2,696 | 608 | 80.3% |
| Partially-Flooded | 440 | 527 | 1,820 | 65.3% |

誤りは `Flooded` ↔ `Partially-Flooded` の組（606 + 440）に集中しています。分類器には **背景クラスがない**ため、検出器のゴミボックスも 3 クラスのいずれかに強制的に割り当てられます。

### 計数

| チェックポイント | 計数ボックス @ 0.30 | 正解 | 差 | MAE / 家屋のある画像 |
| --- | --- | --- | --- | --- |
| `best_ep4` | 4,287 | 3,441 | +846（+24.6%） | 10.95（p90 18、最大 49） |
| epoch_20 | 3,414 | 3,441 | −27（−0.8%） | 9.47 |

epoch_20 の総数は正解にほぼ一致しますが、それは **誤差の相殺** です。ある画像は過大に、別の画像は過小に数えており、1 枚あたりの MAE は依然 10 前後です。したがって総数だけを切り離して読むべきではありません。

## 指標の定義

| 指標 | 定義 / 範囲 |
| --- | --- |
| AP | COCO bbox、IoU 0.50 : 0.05 : 0.95 で平均 |
| AP50 / AP75 | IoU 0.50 / 0.75 |
| AP_s / AP_m / AP_l | 面積 < 32² / 32²–96² / > 96² px² のボックス |
| 精度（分類器） | 正解数 / 総数、検証セットの 8,381 クロップで測定 |
| エンドツーエンド | 分類器を検出器のボックスに適用、そのボックスが正解と IoU 0.5 で一致 |
| MAE | 家屋のある画像における \|計数 − 正解\| の平均 |

AP は `score_thr` 0.05 で測定し、計数はしきい値 0.30 を使います。スケールが異なるため、AP と精度を直接比較しないでください。

## 検証

Web アプリ自身のコード（`best_ep4`、しきい値 0.30）で 203 枚を再実行すると 4,288 ボックスとなり、4,287 に対して **+1 ボックス（0.023%）** の差でした。`Flooded` クラスは 890/890 で完全一致し、差は Colab（Tesla T4）と Web 実行マシン（RTX 2050）の OpenCV および GPU アーキテクチャの違いに由来します。4,287 という基準値は、この検証を再現できるよう `kiem_tra_moi_truong.py` に `MOC_4287` としてハードコードされています。

詳細: `nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt`.

## リポジトリ内のドキュメント

| ファイル | 内容 |
| --- | --- |
| [`HUONG_DAN_CHO_BAN.txt`](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.txt) | インストールと実行の完全な手順、よくあるエラーの表付き |
| [`KET_LUAN_KIEM_CHUNG.txt`](nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt) | 4,287 ボックス検証の記録と +1 の差の説明 |
| [`01_thong_ke_du_lieu.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt) | 2,778 枚 · 44,967 ボックスの分割別統計 |
| [`02_ket_qua_detector.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/02_ket_qua_detector.txt) | epoch_20 チェックポイントの指標別 AP |
| [`03_ket_qua_classifier.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/03_ket_qua_classifier.txt) | 分類器の混同行列と精度 |
| [`04_dem_nha_test.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/04_dem_nha_test.txt) | epoch_20 チェックポイントのクラス別計数表 |

## ディレクトリ構成

```
.
├── export/                              # アプリが出力した結果: png/, jpg/, csv/
├── nha_ngap_v002_code/nha_ngap_v002_code/
│   ├── huan_luyen/                      # Colab ノートブック、検出器設定、結果ファイル 4 つ
│   └── web/                             # 旧 Web 版、比較用に保持
└── nha_ngap_web/nha_ngap_web/           # メインアプリケーション
    ├── ung_dung.py                      # FastAPI: 画面とモデルのライフサイクル
    ├── loi_suy_luan.py                  # 純粋な推論、Web に依存しない
    ├── kiem_tra_moi_truong.py           # 5 段階の環境チェック、--dem-test フラグ付き
    ├── giao_dien/index.html
    ├── checkpoints/                     # cau_hinh_detector.py（同梱）; .pth 2 つは自分で追加
    └── test/                            # テスト画像 203 枚
```

## インストールと実行

### 動作要件

| 項目 | 値 |
| --- | --- |
| OS | Windows 10/11 64 ビット |
| Python | **3.13 64 ビット**、python.org から導入 — Microsoft Store 版は不可、3.12 と 3.14 では `mmcv` を導入できない |
| GPU | NVIDIA、RTX 2050 4 GB で検証済み |
| ドライバー | R580 以降（CUDA 13.2） |
| ディスク | 約 6–7 GB、初回インストールで約 2.5 GB をダウンロード |
| パス | ASCII、アクセント記号と空白を含まないこと |

### チェックポイント

リポジトリに `.pth` ファイルは **含まれていません**（`.gitignore` が `**/*.pth` を除外）。次の 3 ファイルが `nha_ngap_web/nha_ngap_web/checkpoints/` 以下に必要です:

```
work_det/cau_hinh_detector.py                     リポジトリに同梱
work_det/work_aug/best_coco_bbox_mAP_epoch_4.pth  自分で追加
work_cls/convnext_tiny_3lop.pth                   自分で追加
```

3 つのうち 1 つでも欠けると Web アプリは起動しません。`.pth` 2 つは学習したマシン、またはチームのドライブ（`nha_ngap_v002_out/`）から取得します。

### 手順

```bat
cd nha_ngap_web\nha_ngap_web

:: venv を作成し約 2.5 GB をダウンロード、10〜30 分
CAI_DAT.bat

:: 5 段階すべてを通過すること
venv\Scripts\python.exe kiem_tra_moi_truong.py

:: http://127.0.0.1:8000 を開く
CHAY_WEB.bat
```

ポート 8000 が使用中の場合は `CHAY_WEB.bat --cong 8080` を使ってください。

報告書の計数値を再現する場合（2〜3 分、約 4,287 ボックスになるはず）:

```bat
venv\Scripts\python.exe kiem_tra_moi_truong.py --dem-test
```

Web アプリには 4 つのエンドポイントがあります: `GET /`、`GET /api/thong_tin`、`POST /api/du_doan`、`POST /api/export`（PNG/JPG/CSV を `export/` に保存）。モデルは起動時に一度だけ読み込まれ、`mmdet` がスレッドセーフでなく GPU も 4 GB しかないため、リクエストは並列ではなくキューで処理されます。

## 既知の制約

- **テストセットが小さく単一ソース** — 203 枚、外部テストなし。
- **ラベルの曖昧さ** — `Flooded` と `Partially-Flooded` が分類器の誤りの大半を占める。
- **分割にソースの重複が残る** — MD5 グループ 7 件、基底ファイル名 49 件、動画グループ 3 件。重複排除とソース単位の分割が必要。
- **総数が合っても 1 枚ごとに合うわけではない** — MAE 10.95、p90 18、最大 49。
- **背景クラスがない** — ゴミボックスが 3 クラスのいずれかに割り当てられ、過大計数を招く。
- **しきい値が未最適化** — スコアが未校正で、しきい値を上げても精度と再現率の交換になるだけで原因は解決しない。
- **小さな家屋** — 最小アンカーと単一スケール学習により AP_s が低い。
- **本番利用には不十分** — 実際の計数前に背景クラス、校正、クリーンな分割が必要。

## 改善の方向性

| 優先度 | 作業 | 理由 |
| --- | --- | --- |
| P0 | 分類器に背景／棄却クラスを追加 | ゴミボックスが現在は家屋として扱われる — 過大計数の最大要因 |
| P0 | 検証セットでスコアを校正 | 現在の 0.30 というしきい値に確率的な意味はない |
| P1 | F1 ではなく計数 MAE でしきい値を選ぶ | 目的が検出ではなく計数だから |
| P1 | 小さな家屋向けのアンカー / タイル分割 / ランダムスケール | AP_s が 0.169 にとどまる |
| P2 | 重複排除とソース単位の分割 | テストがクリーンになって初めて数値を完全に信頼できる |
| P2 | `max_per_img` を 150–200 に引き上げ | 203 枚中 2 枚が 100 ボックスの上限に掛かっている |

## ライセンス

- **ソースコード**: MIT — [`LICENSE`](LICENSE) を参照。
- **データ**: Roboflow の `merged_dataset` v1、CC BY 4.0 — 再利用時は出典表示が必須。

---

*[README.md](README.md) からの翻訳です。数値に差異がある場合はベトナム語版が正です。*
