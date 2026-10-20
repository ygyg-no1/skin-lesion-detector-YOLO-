# 皮肤病变检测系统（Skin Lesion Detector）

一个可直接运行的演示系统：**FastAPI 后端 + 单页前端**，基于 YOLOv8m 做六类皮肤病变目标检测，并可选择性调用 DeepSeek 大模型对检测结果生成文字解读。

> ⚠️ 仅供科研与教学演示，**不是医疗器械，不能用于临床诊断**。

## 功能

- 账号注册 / 登录（JWT 鉴权）
- 上传皮肤镜图片 → YOLO 目标检测（6 类）→ 返回类别、置信度、包围框与标注图
- 检测历史：保存 / 列表 / 详情 / 删除
- 可选：调用 DeepSeek 对检测结果生成"病理分析"文字（不配置密钥也能正常检测）

## 目录结构

```
backend/main.py          FastAPI 应用（接口、鉴权、推理、历史）
frontend/index.html      单页前端（Tailwind CDN + Chart.js）
models/best.pt           YOLOv8m 六类权重（fp16，约 50 MB）
models/README.md         模型卡
requirements.txt         Python 依赖
.env.example             环境变量模板
scripts/start_backend.*  一键启动脚本（sh / bat）
docs/                    数据与训练说明
```

## 快速开始

1. **准备环境**：Python 3.10+，建议使用虚拟环境。

2. **安装依赖**

```bash
pip install -r requirements.txt
# PyTorch 请按本机 CUDA 版本单独安装，例如 CUDA 11.8：
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
```

3. **配置环境变量**

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

至少建议设置 `SECRET_KEY`；如需 AI 解读再填 `DEEPSEEK_API_KEY`。

4. **启动后端**

```bash
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000
# 或用脚本：scripts/start_backend.sh（Windows 用 scripts/start_backend.bat）
```

首次启动会在 `backend/` 下自动创建 `uploads/`、`history/`、`users/`（运行时数据，已被 `.gitignore` 排除）。
启动成功后可访问 <http://localhost:8000/docs> 查看自动生成的接口文档。

5. **打开前端**

直接用浏览器打开 `frontend/index.html`；若浏览器限制本地文件访问，可在 `frontend/` 下起一个静态服务器：

```bash
cd frontend && python -m http.server 8080   # 然后访问 http://localhost:8080
```

## 配置项

| 变量 | 说明 | 默认值 |
|---|---|---|
| `SECRET_KEY` | JWT 签名密钥，**生产环境必须改成随机长串** | 开发占位值（会打印警告） |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥；留空则"AI 解读"不可用 | 空 |
| `DEEPSEEK_API_URL` | DeepSeek 接口地址 | 官方 chat/completions |
| `DEEPSEEK_MODEL` | 模型名 | `deepseek-chat` |
| `LESION_MODEL` | YOLO 权重路径 | `../models/best.pt` |
| `ALLOW_ORIGINS` | CORS 白名单，逗号分隔 | `*` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 登录 token 有效期（分钟） | `30` |

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/register` | 注册（需要邮箱验证码流程，见代码） |
| POST | `/api/token` | 登录，返回 JWT |
| GET | `/api/users/me` | 当前用户信息 |
| POST | `/api/detect/image` | 上传图片并检测（multipart） |
| POST | `/api/analyze/pathology` | 对检测结果做 DeepSeek 文字解读（可选） |
| GET | `/api/history` | 检测历史列表 |
| GET / DELETE | `/api/history/{id}` | 历史详情 / 删除 |

## 模型

- 架构：YOLOv8m（detect），6 类
- 类别：`vascular_lesion`、`basal_cell_carcinoma`、`actinic_keratosis`、`dermatofibroma`、`melanoma`、`nevus`
- 训练：imgsz 640、150 epochs、batch 10、AdamW、lr0 0.001
- 权重约 50 MB（fp16，已剥离优化器与 EMA，只保留推理权重）

详见 [`models/README.md`](models/README.md)。

## 数据

训练数据为 ISIC 皮肤镜图像，共 **616 张**，按类别分目录：

| 类别 | 张数 |
|---|---|
| basal_cell_carcinoma | 321 |
| melanoma | 87 |
| actinic_keratosis | 80 |
| nevus | 71 |
| dermatofibroma | 42 |
| vascular_lesion | 15 |

因体积（约 162 MB）与授权原因，**图像不随仓库分发**。复现训练请从 [ISIC Archive](https://www.isic-archive.com/) 下载后按 YOLO 格式组织，详见 [`docs/数据与训练说明.md`](docs/数据与训练说明.md)。

## 安全与隐私说明

- 原始工程中的真实用户数据（账号、邮箱、上传的病灶图片、检测历史）**已全部移除**，本仓库不含任何个人数据。
- 原代码里硬编码的 DeepSeek 密钥与绝对路径已清除，改为环境变量 / 相对路径。
- 演示环境的账号、上传图、历史记录以明文存放在本地目录，仅适合单机或内网使用；请勿直接暴露到公网。
- 若你在别处使用过本项目的旧版本，请到 DeepSeek 控制台**吊销并重新生成**密钥。

## 许可与免责

- 本项目代码采用 MIT 许可，见 [`LICENSE`](LICENSE)。
- 依赖的 [Ultralytics](https://github.com/ultralytics/ultralytics) 为 AGPL-3.0 许可，请遵守其条款。
- 本项目仅用于研究与教学演示，输出结果不构成任何医疗建议或诊断依据。
