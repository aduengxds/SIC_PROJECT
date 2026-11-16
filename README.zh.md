# 从航拍图像中识别与统计受淹房屋

[Tiếng Việt](README.md) · [English](README.en.md) · **中文** · [Français](README.fr.md) · [한국어](README.ko.md) · [日本語](README.ja.md)

## 团队

| 成员 | GitHub |
| --- | --- |
| Phùng Duy Hoàng（冯维黄 · Féng Wéi Huáng） | [@21xDHoang](https://github.com/21xDHoang) |
| Bùi Nam Khánh（裴南庆 · Péi Nán Qìng） | [@NK-Lync](https://github.com/NK-Lync) |
| Tào Hoàng Minh Vũ（曹黄明武 · Cáo Huáng Míng Wǔ） | [@aduengxds](https://github.com/aduengxds) |

*括号内为汉语读法；正式写法以越南语原名为准。*

两阶段流水线：**检测器**定位航拍图像中的每一栋房屋，**分类器**判读裁剪区域的状态，然后按类别统计房屋数量。

- 检测器：**Cascade R-CNN + ResNet-50-FPN**，单一 `house` 类别
- 分类器：**ConvNeXt-Tiny**，三个类别 `Flooded` / `Not-Flooded` / `Partially-Flooded`
- Web 应用：**FastAPI**，完全在本机 GPU 上运行

| 项目 | 指标 | 数值 |
| --- | --- | --- |
| 检测器（`best_ep4`） | AP50 | 0.631 |
| 分类器 | 端到端准确率 | 0.6682 |
| 计数（阈值 0.30） | 4,287 个框 / 3,441 个标注框 | +24.6% |

> 默认计数阈值 0.30 偏向召回率，因此总数**多计 24.6%**，而单张图像上的误差仍然很大（MAE 10.95；p90 为 18；最大 49）。该流水线适合做粗略普查，**尚不足以精确统计单张图像**——见 *已知限制*。

## 问题定义

| | |
| --- | --- |
| 输入 | 一张 640×640 的航拍图像 |
| 输出 | 每栋房屋一个边界框，各带三种状态之一，以及按类别的计数表 |

主要难点：房屋非常小、水体浑浊、`Flooded` 与 `Partially-Flooded` 标注界限模糊，图像可能完全为空或极其密集。

## 数据

来源 Roboflow，图像 640×640，三个类别 `Flooded`、`Not-Flooded`、`Partially-Flooded`。合计 **2,778 张图像 · 44,967 个框**；数据划分原样取自 Roboflow，测试集只在报告时使用一次。

| 属性 | 取值 |
| --- | --- |
| 数据集 | [`merged_dataset` – v1](https://universe.roboflow.com/housedetectionclassification-jugwz/merged_dataset-dk0ch) |
| 许可证 | **CC BY 4.0**——再次使用时必须署名 |
| 标注格式 | COCO |
| Roboflow 预处理 | 自动转正（去除 EXIF）、拉伸缩放至 640×640；无数据增强 |
| 导出日期 | 2024-11-07 |

| 划分 | 图像 | 框 | 说明 |
| --- | --- | --- | --- |
| Train | 2,116 | 33,145 | 小于 32 px 的框占 10.7%；小于 64 px 占 48.5% |
| Valid | 459 | 8,381 | 用于选择模型与阈值 |
| Test | 203 | 3,441 | 其中 12 张图像没有房屋 |

完整统计：`nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt`。

## 架构

两个模型学习不同的任务：检测器只回答"房屋在哪里"，分类器只回答"是什么状态"。推理时，裁剪区域来自检测器输出的框，不使用真实标注。

**检测器**——Cascade R-CNN，骨干网络 ResNet-50-FPN，3 级级联，IoU 分别为 0.5 / 0.6 / 0.7，一个 `house` 类加背景（2 个 logit），锚框 32–512 px（stride 4–64 × scale 8），`keep_ratio` 缩放流水线 1333×800，没有 `RandomResize`。配置：`huan_luyen/cau_hinh_detector.py`。

**分类器**——ConvNeXt-Tiny（ImageNet-1K V1），27.82 M 参数，输出 768 → 3 类，输入 224 px，训练与推理都使用 15% 的外扩边距，12 个 epoch，AdamW 3e-4，label smoothing 0.05。

默认流水线：图像 → 检测器 → 保留 score ≥ 0.30、最多 100 个框 → 裁剪（+15%）→ ConvNeXt-Tiny → 按类别计数。

## 重新训练

两个 notebook 可在 Colab 上运行（T4 即可），打开后自上而下点击 *Run all*：

- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002_Aug.ipynb`——带数据增强的版本，产出当前使用的 `best_ep4`
- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002.ipynb`——不带增强的版本；其 epoch 20 的数值在 `02_ket_qua_detector.txt` 中

| 单元 | 内容 |
| --- | --- |
| 1–3 | 检查环境，安装 `mmcv` / `mmengine` / `mmdet`，验证 CUDA 算子确实可用 |
| 4–5 | 获取数据、统计，并转换为两种格式：检测器用的单一 `house` 类，以及分类器用的原始三类 |
| 6–7 | 训练 Cascade R-CNN 检测器，绘制学习曲线，在测试集上评估 mAP |
| 8 | 用真实标注框裁剪的样本训练 ConvNeXt-Tiny 分类器 |
| 9–10 | 在 203 张图像上执行 检测 → 分类 → 计数，汇总并写出 4 个结果文件 |
| 11 | Gradio 网页演示 |

检查点按验证集 mAP 以 `save_best` 保存，因此 `best_ep4` 是 **第 4 个 epoch**，不是最后一个 epoch——不要与不带增强版本的 `epoch_20` 混淆。

## 结果

![航拍图像上的识别结果——红色框为受淹，橙色为部分受淹，绿色为未受淹](export/mau_ket_qua.jpg)

*图像由 Web 应用自身导出（`export/mau_ket_qua.jpg`）。框的颜色遵循 `loi_suy_luan.py`：红色 = `Flooded`，橙色 = `Partially-Flooded`，绿色 = `Not-Flooded`。*

### 检测器

有两个检查点的结果，**不要混用它们的数值**：

| 检查点 | AP | AP50 | AP75 | AP_s | 来源 |
| --- | --- | --- | --- | --- | --- |
| `best_ep4`（Web 应用默认） | 0.313 | 0.631 | 0.284 | 0.169 | 报告中给出的数值 |
| epoch_20（无增强） | 0.277 | 0.584 | 0.233 | 0.149 | `02_ket_qua_detector.txt` |

AP_s 偏低是真实的局限：RPN 最小的锚框为 32 px（stride 4 × scale 8），且训练只使用单一缩放流水线，因此只有几十像素的房屋没有匹配良好的锚框。

### 分类器

在验证集 8,381 个裁剪样本上：**准确率 0.7195**。在测试集裁剪样本（真实标注）上：0.6713。使用检测器的框时：**端到端 0.6682**——几乎不下降，说明检测器的裁剪并非瓶颈。

| 真实 \ 预测 | Flooded | Not-Flooded | Partially-Flooded | 召回率 |
| --- | --- | --- | --- | --- |
| Flooded | 1,514 | 115 | 606 | 67.7% |
| Not-Flooded | 55 | 2,696 | 608 | 80.3% |
| Partially-Flooded | 440 | 527 | 1,820 | 65.3% |

误差集中在 `Flooded` ↔ `Partially-Flooded` 这一对（606 + 440）。分类器**没有背景类**，因此检测器产生的垃圾框仍会被强制归入三类之一。

### 计数

| 检查点 | 计数框 @ 0.30 | 标注 | 偏差 | MAE / 有房屋的图像 |
| --- | --- | --- | --- | --- |
| `best_ep4` | 4,287 | 3,441 | +846（+24.6%） | 10.95（p90 18，最大 49） |
| epoch_20 | 3,414 | 3,441 | −27（−0.8%） | 9.47 |

epoch_20 的总数看似与标注接近，但那是**误差相互抵消**：有的图像多计、有的少计，单张图像的 MAE 仍接近 10。因此不要单独解读总数。

## 指标定义

| 指标 | 定义 / 范围 |
| --- | --- |
| AP | COCO bbox，在 IoU 0.50 : 0.05 : 0.95 上取平均 |
| AP50 / AP75 | IoU 0.50 / 0.75 |
| AP_s / AP_m / AP_l | 框面积 < 32² / 32²–96² / > 96² px² |
| 准确率（分类器） | 正确数 / 总数，在验证集 8,381 个裁剪样本上测得 |
| 端到端 | 分类器运行在检测器框上，该框与标注重合 IoU 0.5 |
| MAE | 有房屋图像上 \|计数 − 标注数\| 的平均值 |

AP 在 `score_thr` 0.05 下测量，而计数使用 0.30 阈值——两者量纲不同，不要把 AP 与准确率直接比较。

## 验证

用 Web 应用自身的代码（`best_ep4`，阈值 0.30）重跑全部 203 张图像，得到 4,288 个框，相对 4,287 **仅差 +1 个框 = 0.023%**。`Flooded` 类完全一致（890/890）；差异来自 Colab（Tesla T4）与本机（RTX 2050）之间 OpenCV 和 GPU 架构的不同。4,287 这一基准被硬编码为 `kiem_tra_moi_truong.py` 中的 `MOC_4287`，以便该检验可复现。

详情：`nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt`。

## 仓库内的文档

| 文件 | 内容 |
| --- | --- |
| [`HUONG_DAN_CHO_BAN.txt`](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.txt) | 完整的安装与运行指南，附常见错误对照表 |
| [`KET_LUAN_KIEM_CHUNG.txt`](nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt) | 4,287 框检验的记录，并解释 +1 的差异 |
| [`01_thong_ke_du_lieu.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt) | 2,778 张图像 · 44,967 个框按划分的统计 |
| [`02_ket_qua_detector.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/02_ket_qua_detector.txt) | epoch_20 检查点的各项 AP |
| [`03_ket_qua_classifier.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/03_ket_qua_classifier.txt) | 分类器的混淆矩阵与准确率 |
| [`04_dem_nha_test.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/04_dem_nha_test.txt) | epoch_20 检查点的按类别计数表 |
| [`HUONG_DAN_CHO_BAN.en.txt`](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.en.txt) | 安装指南的英文译本 |

## 目录结构

```
.
├── export/                              # 应用导出的结果：png/、jpg/、csv/
├── nha_ngap_v002_code/nha_ngap_v002_code/
│   ├── huan_luyen/                      # Colab notebook、检测器配置、4 个结果文件
│   └── web/                             # 旧的 Web 副本，保留用于对照
└── nha_ngap_web/nha_ngap_web/           # 主应用
    ├── ung_dung.py                      # FastAPI：界面与模型生命周期
    ├── loi_suy_luan.py                  # 纯推理，不涉及 Web
    ├── kiem_tra_moi_truong.py           # 5 步环境检查，带 --dem-test 参数
    ├── giao_dien/index.html
    ├── checkpoints/                     # cau_hinh_detector.py（已含）；2 个 .pth 需自行添加
    └── test/                            # 203 张测试图像
```

## 安装与运行

### 环境要求

| 项目 | 取值 |
| --- | --- |
| 操作系统 | Windows 10/11 64 位 |
| Python | **3.13 64 位**，从 python.org 下载——不要用 Microsoft Store 版本；3.12 与 3.14 无法安装 `mmcv` |
| GPU | NVIDIA；已在 RTX 2050 4 GB 上验证 |
| 驱动 | R580 或更新（CUDA 13.2） |
| 磁盘 | 约 6–7 GB；首次安装需下载约 2.5 GB |
| 路径 | ASCII，不含音标符号，不含空格 |

### 检查点

仓库**不包含** `.pth` 文件（`.gitignore` 排除 `**/*.pth`）。以下三个文件必须齐全，放在 `nha_ngap_web/nha_ngap_web/checkpoints/` 下：

```
work_det/cau_hinh_detector.py                     仓库中已含
work_det/work_aug/best_coco_bbox_mAP_epoch_4.pth  自行添加
work_cls/convnext_tiny_3lop.pth                   自行添加
```

三者缺一则 Web 应用无法启动。两个 `.pth` 文件取自训练机器或团队网盘（`nha_ngap_v002_out/`）。

### 步骤

```bat
cd nha_ngap_web\nha_ngap_web

:: 创建 venv 并下载约 2.5 GB，耗时 10-30 分钟
CAI_DAT.bat

:: 必须通过全部 6 个步骤
venv\Scripts\python.exe kiem_tra_moi_truong.py

:: 打开 http://127.0.0.1:8000
CHAY_WEB.bat
```

若 8000 端口被占用，使用 `CHAY_WEB.bat --cong 8080`。

复现报告中的计数数值（2–3 分钟，应得到约 4,287 个框）：

```bat
venv\Scripts\python.exe kiem_tra_moi_truong.py --dem-test
```

Web 应用有四个接口：`GET /`、`GET /api/thong_tin`、`POST /api/du_doan`、`POST /api/export`（将 PNG/JPG/CSV 保存到 `export/`）。模型在启动时只加载一次；由于 `mmdet` 不是线程安全的且 GPU 只有 4 GB，请求会排队而不是并行执行。

### 简短排错

- 安装路径必须是 ASCII（不能有越南语声调符号或空格），否则会出现难以理解的错误。
- 端口 8000 被占用时：`CHAY_WEB.bat --cong 8080`。
- 没有可用的 GPU 时仍可在 CPU 上运行：结果相同，但慢约 50 倍；`--cpu` 可关闭警告。
- 完整指南：[English](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.en.txt) · [Tiếng Việt](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.txt)。

## 已知限制

- **测试集小且来源单一**——203 张图像，没有外部测试集。
- **标注界限模糊**——`Flooded` 与 `Partially-Flooded` 占了分类器误差的大部分。
- **数据划分仍存在来源重叠**——7 组 MD5、49 个基础文件名、3 组视频；需要去重并按来源划分。
- **总数正确不等于单图正确**——MAE 10.95；p90 18；最大 49。
- **缺少背景类**——垃圾框被归入三类之一，导致多计。
- **阈值未优化**——分数未校准；提高阈值只是在精确率与召回率之间取舍，并不能解决根因。
- **小房屋**——AP_s 偏低，源于最小锚框与单一尺度训练。
- **尚不可用于生产**——在真实计数前需要背景类、校准和干净的数据划分。

## 改进方向

| 优先级 | 工作 | 理由 |
| --- | --- | --- |
| P0 | 为分类器增加背景 / 拒识类 | 垃圾框目前被当作房屋——多计的最大来源 |
| P0 | 在验证集上校准分数 | 当前 0.30 阈值不具有概率意义 |
| P1 | 按计数 MAE 而非 F1 选择阈值 | 本任务的目标是计数，不是检测 |
| P1 | 针对小房屋的锚框 / 切片 / 随机缩放 | AP_s 仅为 0.169 |
| P2 | 去重并按来源划分数据 | 测试集更干净，数值才完全可信 |
| P2 | 将 `max_per_img` 提高到 150–200 | 仍有 2/203 张图像被 100 框上限截断 |

## 许可证

- **源代码**：MIT——见 [`LICENSE`](LICENSE)。
- **数据**：Roboflow 的 `merged_dataset` v1，许可证 CC BY 4.0——再次使用时必须署名。

---

*译自 [README.md](README.md)。若数值有出入，以越南语版本为准。*
