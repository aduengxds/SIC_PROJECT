# Nhận diện & đếm nhà ngập từ ảnh flycam

## Nhóm thực hiện

| Thành viên | GitHub |
| --- | --- |
| Phòng Duy Hoàng | [@21xDHoang](https://github.com/21xDHoang) |
| Bùi Nam Khánh | [@NK-Lync](https://github.com/NK-Lync) |
| Tào Hoàng Minh Vũ | [@aduengxds](https://github.com/aduengxds) |

Pipeline hai tầng: **detector** tìm vị trí từng ngôi nhà trong ảnh flycam, **classifier** đọc trạng thái của vùng ảnh vừa cắt ra, rồi đếm số nhà theo từng lớp.

- Detector: **Cascade R-CNN + ResNet-50-FPN**, một lớp `house`
- Classifier: **ConvNeXt-Tiny**, ba lớp `Flooded` / `Not-Flooded` / `Partially-Flooded`
- Ứng dụng web: **FastAPI**, chạy hoàn toàn cục bộ trên máy có GPU

| Hạng mục | Chỉ số | Giá trị |
| --- | --- | --- |
| Detector (`best_ep4`) | AP50 | 0,631 |
| Classifier | accuracy end-to-end | 0,6682 |
| Đếm nhà @ ngưỡng 0,30 | 4.287 box / 3.441 box có nhãn | +24,6 % |

> Ngưỡng đếm mặc định 0,30 ưu tiên recall, nên tổng số box **đếm dư 24,6 %**, trong khi sai số trên từng ảnh vẫn lớn (MAE 10,95; p90 18; max 49). Pipeline phù hợp khảo sát sơ bộ, **chưa đủ để đếm chính xác từng ảnh** — xem mục *Hạn chế đã biết*.

## Bài toán

| | |
| --- | --- |
| Đầu vào | một ảnh flycam 640×640 |
| Đầu ra | box cho từng ngôi nhà, mỗi box mang một trong ba trạng thái, và bảng đếm theo lớp |

Khó khăn chính: nhà rất nhỏ, nước đục, nhãn `Flooded` / `Partially-Flooded` nhập nhằng, ảnh có thể trắng hoặc rất dày nhà.

## Dữ liệu

Nguồn Roboflow, ảnh 640×640, ba lớp `Flooded`, `Not-Flooded`, `Partially-Flooded`. Tổng **2.778 ảnh · 44.967 box**; split lấy nguyên từ Roboflow, tập test chỉ dùng một lần khi báo cáo.

| Thuộc tính | Giá trị |
| --- | --- |
| Dataset | [`merged_dataset` – v1](https://universe.roboflow.com/housedetectionclassification-jugwz/merged_dataset-dk0ch) |
| Giấy phép | **CC BY 4.0** — dùng lại phải ghi nguồn |
| Định dạng nhãn | COCO |
| Tiền xử lý của Roboflow | Auto-orient (bỏ EXIF), resize 640×640 stretch; không augmentation |
| Ngày export | 07/11/2024 |

| Tập | Ảnh | Box | Ghi chú |
| --- | --- | --- | --- |
| Train | 2.116 | 33.145 | box < 32 px chiếm 10,7 %; < 64 px chiếm 48,5 % |
| Valid | 459 | 8.381 | dùng chọn model và chọn ngưỡng |
| Test | 203 | 3.441 | 12 ảnh không có nhà |

Thống kê đầy đủ: `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt`.

## Kiến trúc

Hai model học hai việc khác nhau: detector chỉ trả lời "nhà ở đâu", classifier chỉ trả lời "trạng thái gì". Lúc suy luận, crop được lấy từ box của detector (không dùng nhãn thật).

**Detector** — Cascade R-CNN, backbone ResNet-50-FPN, 3 stage với IoU 0.5 / 0.6 / 0.7, một lớp `house` + background (2 logit), anchor 32–512 px (stride 4–64 × scale 8), pipeline resize `keep_ratio` 1333×800, không có `RandomResize`. Config: `huan_luyen/cau_hinh_detector.py`.

**Classifier** — ConvNeXt-Tiny (ImageNet-1K V1), 27,82 M tham số, đầu ra 768 → 3 lớp, ảnh vào 224 px, crop mở rộng 15 % margin ở cả train và inference, 12 epoch AdamW 3e-4, label smoothing 0,05.

Pipeline mặc định: ảnh → detector → lọc score ≥ 0,30, tối đa 100 box → crop (+15 %) → ConvNeXt-Tiny → đếm theo lớp.

## Huấn luyện lại

Hai notebook chạy trên Colab (T4 là đủ), mở rồi bấm *Run all* từ trên xuống:

- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002_Aug.ipynb` — bản có augmentation, sinh ra `best_ep4` đang dùng
- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002.ipynb` — bản không augmentation; số của epoch 20 nằm trong `02_ket_qua_detector.txt`

| Ô | Việc |
| --- | --- |
| 1–3 | kiểm tra môi trường, cài `mmcv` / `mmengine` / `mmdet`, kiểm tra CUDA ops chạy thật |
| 4–5 | lấy dữ liệu, thống kê, chuyển sang hai định dạng: một lớp `house` cho detector và ba lớp gốc cho classifier |
| 6–7 | train detector Cascade R-CNN, vẽ đường cong học, đánh giá mAP trên test |
| 8 | train classifier ConvNeXt-Tiny trên crop của box thật |
| 9–10 | chạy detect → classify → đếm trên 203 ảnh, tổng kết và ghi ra 4 file kết quả |
| 11 | web Gradio để demo |

Checkpoint lưu theo `save_best` trên valid mAP, nên `best_ep4` là **epoch 4**, không phải epoch cuối — đừng nhầm với `epoch_20` của bản không augmentation.

## Kết quả

![Kết quả nhận diện trên ảnh flycam — khung đỏ là ngập, cam là ngập một phần, xanh là không ngập](export/mau_ket_qua.jpg)

*Ảnh do chính bản web xuất ra (`export/mau_ket_qua.jpg`). Màu khung theo `loi_suy_luan.py`: đỏ = `Flooded`, cam = `Partially-Flooded`, xanh = `Not-Flooded`.*

### Detector

Có kết quả cho hai checkpoint, **đừng trộn số của chúng**:

| Checkpoint | AP | AP50 | AP75 | AP_s | Nguồn |
| --- | --- | --- | --- | --- | --- |
| `best_ep4` (bản web mặc định) | 0,313 | 0,631 | 0,284 | 0,169 | số trình bày trong báo cáo |
| epoch_20 (no-aug) | 0,277 | 0,584 | 0,233 | 0,149 | `02_ket_qua_detector.txt` |

AP_s thấp là hạn chế thật: anchor nhỏ nhất của RPN là 32 px (stride 4 × scale 8) và toàn bộ quá trình train chỉ dùng một pipeline resize, nên nhà chỉ vài chục pixel khó có anchor phù hợp.

### Classifier

Trên 8.381 crop của tập valid: **accuracy 0,7195**. Trên test crop (nhãn thật): 0,6713. Khi dùng box của detector: **end-to-end 0,6682** — gần như không giảm, tức crop của detector không phải nút thắt.

| Nhãn thật \ Dự đoán | Flooded | Not-Flooded | Partially-Flooded | Recall |
| --- | --- | --- | --- | --- |
| Flooded | 1.514 | 115 | 606 | 67,7 % |
| Not-Flooded | 55 | 2.696 | 608 | 80,3 % |
| Partially-Flooded | 440 | 527 | 1.820 | 65,3 % |

Lỗi tập trung ở cặp `Flooded` ↔ `Partially-Flooded` (606 + 440). Classifier **không có lớp nền**, nên box rác từ detector vẫn bị gán vào một trong ba lớp.

### Đếm nhà

| Checkpoint | Box đếm @ 0,30 | Nhãn | Lệch | MAE / ảnh có nhà |
| --- | --- | --- | --- | --- |
| `best_ep4` | 4.287 | 3.441 | +846 (+24,6 %) | 10,95 (p90 18, max 49) |
| epoch_20 | 3.414 | 3.441 | −27 (−0,8 %) | 9,47 |

Tổng của epoch_20 gần khớp nhãn nhưng đó là **triệt tiêu sai số**: ảnh này đếm dư, ảnh kia đếm thiếu, nên MAE từng ảnh vẫn gần 10. Vì vậy không nên đọc riêng con số tổng.

## Định nghĩa chỉ số

| Chỉ số | Định nghĩa / phạm vi |
| --- | --- |
| AP | COCO bbox, lấy trung bình qua IoU 0.50 : 0.05 : 0.95 |
| AP50 / AP75 | IoU 0.50 / 0.75 |
| AP_s / AP_m / AP_l | box có diện tích < 32² / 32²–96² / > 96² px² |
| Accuracy (classifier) | đúng / tổng, đo trên 8.381 crop của tập valid |
| End-to-end | classifier chạy trên box của detector khớp nhãn ở IoU 0.5 |
| MAE | trung bình \|số đếm − số nhãn\| trên ảnh có nhà |

AP đo ở `score_thr` 0,05 còn đếm dùng ngưỡng 0,30 — hai con số khác thang, đừng so trực tiếp AP với accuracy.

## Kiểm chứng

Chạy lại toàn bộ 203 ảnh bằng chính code của bản web (`best_ep4`, ngưỡng 0,30) cho 4.288 box, lệch **+1 box trên 4.287 = 0,023 %**. Lớp `Flooded` khớp tuyệt đối 890/890; phần lệch nằm ở khác biệt OpenCV và kiến trúc GPU giữa Colab (Tesla T4) và máy chạy web (RTX 2050). Mốc 4.287 được hard-code thành `MOC_4287` trong `kiem_tra_moi_truong.py` để phép thử này tái lập được.

Chi tiết: `nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt`.

## Tài liệu trong repo

| Tệp | Nội dung |
| --- | --- |
| [`HUONG_DAN_CHO_BAN.txt`](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.txt) | Hướng dẫn cài đặt và chạy đầy đủ, kèm bảng lỗi hay gặp |
| [`KET_LUAN_KIEM_CHUNG.txt`](nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt) | Biên bản phép thử 4.287 box, giải thích chỗ lệch +1 |
| [`01_thong_ke_du_lieu.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt) | Thống kê 2.778 ảnh · 44.967 box theo từng tập |
| [`02_ket_qua_detector.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/02_ket_qua_detector.txt) | AP từng chỉ số của checkpoint epoch_20 |
| [`03_ket_qua_classifier.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/03_ket_qua_classifier.txt) | Confusion matrix và accuracy của classifier |
| [`04_dem_nha_test.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/04_dem_nha_test.txt) | Bảng đếm theo lớp của checkpoint epoch_20 |

## Cấu trúc thư mục

```
.
├── export/                              # kết quả app xuất ra: png/, jpg/, csv/
├── nha_ngap_v002_code/nha_ngap_v002_code/
│   ├── huan_luyen/                      # notebook Colab, config detector, 4 file kết quả
│   └── web/                             # bản web cũ, giữ để đối chiếu
└── nha_ngap_web/nha_ngap_web/           # ứng dụng chính
    ├── ung_dung.py                      # FastAPI: giao diện và vòng đời model
    ├── loi_suy_luan.py                  # suy luận thuần, không biết gì về web
    ├── kiem_tra_moi_truong.py           # 5 bước kiểm tra môi trường, có cờ --dem-test
    ├── giao_dien/index.html
    ├── checkpoints/                     # cau_hinh_detector.py (có sẵn); 2 file .pth (tự thêm)
    └── test/                            # 203 ảnh kiểm thử
```

## Cài đặt & chạy

### Yêu cầu

| Hạng mục | Giá trị |
| --- | --- |
| Hệ điều hành | Windows 10/11 64-bit |
| Python | **3.13 64-bit**, tải từ python.org — không dùng bản Microsoft Store; 3.12 và 3.14 không cài được `mmcv` |
| GPU | NVIDIA; đã kiểm chứng trên RTX 2050 4 GB |
| Driver | R580 trở lên (CUDA 13.2) |
| Dung lượng | ~6–7 GB; lần cài đầu tải ~2,5 GB |
| Đường dẫn | ASCII, không dấu, không khoảng trắng |

### Checkpoint

Repo **không kèm** file `.pth` (`.gitignore` loại `**/*.pth`). Cần đủ ba tệp trong `nha_ngap_web/nha_ngap_web/checkpoints/`:

```
work_det/cau_hinh_detector.py                     có sẵn trong repo
work_det/work_aug/best_coco_bbox_mAP_epoch_4.pth  tự thêm
work_cls/convnext_tiny_3lop.pth                   tự thêm
```

Thiếu một trong ba là web không chạy được. Hai file `.pth` lấy từ máy đã huấn luyện hoặc từ Drive của nhóm (`nha_ngap_v002_out/`).

### Các bước

```bat
cd nha_ngap_web\nha_ngap_web

:: tạo venv, tải ~2,5 GB, mất 10-30 phút
CAI_DAT.bat

:: phải qua đủ 5 bước
venv\Scripts\python.exe kiem_tra_moi_truong.py

:: mở http://127.0.0.1:8000
CHAY_WEB.bat
```

Cổng 8000 bận thì dùng `CHAY_WEB.bat --cong 8080`.

Kiểm chứng lại con số trong báo cáo (2–3 phút, phải ra khoảng 4.287 box):

```bat
venv\Scripts\python.exe kiem_tra_moi_truong.py --dem-test
```

Web có bốn endpoint: `GET /`, `GET /api/thong_tin`, `POST /api/du_doan`, `POST /api/export` (lưu PNG/JPG/CSV vào `export/`). Model nạp một lần lúc khởi động; vì `mmdet` không thread-safe và GPU chỉ 4 GB, các request được xếp hàng thay vì chạy song song.

## Hạn chế đã biết

- **Tập test nhỏ, một nguồn** — 203 ảnh, chưa có external test.
- **Nhãn nhập nhằng** — `Flooded` và `Partially-Flooded` chiếm phần lớn sai số của classifier.
- **Split còn trùng nguồn** — 7 nhóm MD5, 49 tên cơ sở, 3 nhóm video; cần dedupe và chia theo nguồn.
- **Đếm đúng tổng không có nghĩa là đúng từng ảnh** — MAE 10,95; p90 18; max 49.
- **Thiếu lớp nền** — box rác bị gán thành một trong ba lớp, gây đếm dư.
- **Ngưỡng chưa tối ưu** — score chưa calibrate; nâng ngưỡng chỉ đổi precision/recall chứ không sửa gốc.
- **Nhà nhỏ** — AP_s thấp do anchor nhỏ nhất và chỉ train một scale.
- **Chưa dùng được cho production** — cần background class, calibration và split sạch trước khi đếm thực tế.

## Hướng cải thiện

| Ưu tiên | Việc | Vì sao |
| --- | --- | --- |
| P0 | Thêm lớp nền / reject class cho classifier | box rác đang bị gán thành nhà — nguồn đếm dư lớn nhất |
| P0 | Calibrate score trên tập valid | ngưỡng 0,30 hiện không mang nghĩa xác suất |
| P1 | Chọn ngưỡng theo count MAE thay vì F1 | mục tiêu của bài là đếm, không phải phát hiện |
| P1 | Anchor / tiling / random scale cho nhà nhỏ | AP_s chỉ đạt 0,169 |
| P2 | Dedupe và chia split theo nguồn | test sạch hơn thì số liệu mới đáng tin tuyệt đối |
| P2 | Tăng `max_per_img` lên 150–200 | còn 2/203 ảnh bị cap ở 100 box |
