# 模型说明

`best.pt` — 皮肤病变目标检测权重

| 项 | 值 |
|---|---|
| 架构 | YOLOv8m（detect 头） |
| 类别数 | 6 |
| 类别 | `vascular_lesion`, `basal_cell_carcinoma`, `actinic_keratosis`, `dermatofibroma`, `melanoma`, `nevus` |
| 输入尺寸 | 640 × 640 |
| 训练轮数 | 150 epochs，batch 10，AdamW，lr0 = 0.001 |
| 框架 | Ultralytics（训练时版本 8.3.63，2025-10） |
| 文件大小 | 52 MB（fp16） |

## 使用说明

- 权重已剥离优化器状态与 EMA，只保留推理权重，可直接被 `ultralytics.YOLO(...)` 加载。
- 训练数据为 ISIC 皮肤镜图像，**不随仓库分发**（体积与授权原因）；复现训练请自行从 ISIC Archive 下载后按 YOLO 格式组织。
- ⚠️ 本模型面向科研与教学演示，**不是医疗器械，不能用于临床诊断**。预测结果仅供研究参考。
