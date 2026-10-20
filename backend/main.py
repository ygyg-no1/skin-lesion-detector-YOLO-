print("后端开始启动...")
import os
import time
import cv2
import numpy as np
import base64
import json
import re  # 新增：导入re模块用于正则处理
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
import uuid
from ultralytics import YOLO
import requests

from jose import jwt, JWTError

# 初始化FastAPI应用
app = FastAPI(title="医学影像检测与分析API")

# 配置跨域（确保前端可正常请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径（自动创建所需目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
USERS_DIR = os.path.join(BASE_DIR, "users")
for dir_path in [UPLOAD_DIR, HISTORY_DIR, USERS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ---------------------------------------------------------------------------
# 配置：一律从环境变量读取（参考项目根目录的 .env.example）
# ---------------------------------------------------------------------------
try:  # 有 .env 就自动加载，没有也不影响启动
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, "..", ".env"))
except Exception:
    pass

# YOLO 模型：默认使用仓库自带的 models/best.pt（6 类皮肤病变检测）
MODEL_PATH = os.environ.get(
    "LESION_MODEL", os.path.abspath(os.path.join(BASE_DIR, "..", "models", "best.pt"))
)
if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        "[ERROR] 未找到模型权重：%s\n"
        "请确认 models/best.pt 存在，或用环境变量 LESION_MODEL 指定权重路径。" % MODEL_PATH
    )
model = YOLO(MODEL_PATH)
model.predict(np.zeros((640, 640, 3)), conf=0.5)
print("模型加载成功：%s" % MODEL_PATH)
print("可识别类别：%s" % model.names)

# DeepSeek 配置：密钥只从环境变量读取，不要写进代码
DEEPSEEK_CONFIG = {
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "api_url": os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
    "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
}

# 认证配置：生产环境务必用环境变量 SECRET_KEY 注入随机字符串
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-please-change")
if SECRET_KEY.startswith("dev-only"):
    print("[WARN] 未设置 SECRET_KEY，正在使用开发默认值，请勿用于生产环境")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# 密码加密配置（使用pbkdf2_sha256避免bcrypt兼容问题）
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
    pbkdf2_sha256__default_rounds=30000
)


# 密码哈希与验证函数
def get_password_hash(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# OAuth2配置
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")


# 数据模型定义
class User(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = Field(default=None)
    disabled: bool = False


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str | None = Field(default=None)


class DetectionResults(BaseModel):
    results: list
    image_name: str
    annotated_image: str
    total_objects: int
    max_confidence: float
    detect_time_ms: int


# 认证工具函数
def get_user(username: str):
    """从文件中获取用户信息"""
    user_file = os.path.join(USERS_DIR, f"{username}.json")
    if os.path.exists(user_file):
        with open(user_file, "r", encoding="utf-8") as f:
            return UserInDB(**json.load(f))
    return None


def authenticate_user(username: str, password: str):
    """验证用户身份"""
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta = None):
    """生成JWT令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前登录用户（依赖注入）"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="无法验证凭据，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
        user = get_user(username)
        if not user:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """确保用户处于激活状态"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="用户已禁用")
    return current_user


# 工具函数：保存分析历史
def save_history(analysis_data, username):
    """保存用户的病理分析历史记录"""
    user_history_dir = os.path.join(HISTORY_DIR, username)
    os.makedirs(user_history_dir, exist_ok=True)
    history_id = str(uuid.uuid4())
    history_path = os.path.join(user_history_dir, f"{history_id}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "id": history_id,
            "username": username,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **analysis_data
        }, f, ensure_ascii=False, indent=2)
    return history_id


# 1. 注册接口
@app.post("/api/register", response_model=User)
async def register(user: UserCreate):
    """用户注册接口"""
    # 验证用户名长度
    if len(user.username) < 3 or len(user.username) > 20:
        raise HTTPException(status_code=400, detail="用户名长度必须在3-20个字符之间")
    # 验证密码长度
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="密码长度不能少于6个字符")
    # 检查用户名是否已存在
    if get_user(user.username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    # 检查邮箱是否已被使用
    email_exists = any(
        json.load(open(os.path.join(USERS_DIR, f))).get("email") == user.email
        for f in os.listdir(USERS_DIR) if f.endswith(".json")
    )
    if email_exists:
        raise HTTPException(status_code=400, detail="该邮箱已被注册")

    # 创建用户并保存
    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name or "",
        "hashed_password": hashed_password,
        "disabled": False
    }
    with open(os.path.join(USERS_DIR, f"{user.username}.json"), "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

    return User(**{k: v for k, v in user_data.items() if k != "hashed_password"})


# 2. 登录接口（获取令牌）
@app.post("/api/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """用户登录并获取访问令牌"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name or ""
        }
    }


# 3. 获取当前用户信息
@app.get("/api/users/me", response_model=User)
async def get_user_info(current_user: User = Depends(get_current_active_user)):
    """获取当前登录用户的基本信息"""
    return current_user


# 4. 图片检测接口
@app.post("/api/detect/image")
async def detect_image(
        file: UploadFile = File(...),
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        current_user: User = Depends(get_current_active_user)
):
    """医学影像检测接口（需要登录）"""
    try:
        start_time = time.time()
        # 验证文件格式
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            raise HTTPException(status_code=400, detail="仅支持JPG、PNG、BMP格式图片")

        # 保存上传文件
        user_upload_dir = os.path.join(UPLOAD_DIR, current_user.username)
        os.makedirs(user_upload_dir, exist_ok=True)
        image_id = str(uuid.uuid4())
        image_path = os.path.join(user_upload_dir, f"{image_id}{file_ext}")
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # 模型检测
        img = cv2.imread(image_path)
        results = model(img, conf=conf_threshold, iou=iou_threshold)
        annotated_img = results[0].plot()

        # 保存标注图片并转为Base64
        annotated_path = os.path.join(user_upload_dir, f"{image_id}_annotated{file_ext}")
        cv2.imwrite(annotated_path, annotated_img)
        _, buffer = cv2.imencode(file_ext, annotated_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # 整理检测结果
        detection_results = []
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            detection_results.append({
                "class_name": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        # 计算统计数据
        detect_time = int((time.time() - start_time) * 1000)
        max_confidence = max([r["confidence"] for r in detection_results]) if detection_results else 0

        return {
            "status": "success",
            "data": {
                "image_id": image_id,
                "original_filename": file.filename,
                "annotated_image": img_base64,
                "detection_results": detection_results,
                "total_objects": len(detection_results),
                "max_confidence": max_confidence,
                "detect_time_ms": detect_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败：{str(e)}")


# 5. 病理分析接口
@app.post("/api/analyze/pathology")
async def analyze_pathology(
        data: DetectionResults,
        x_api_key: str = Header(None),
        current_user: User = Depends(get_current_active_user)
):
    """基于检测结果进行病理分析（需要登录）"""
    try:
        # 验证API密钥
        api_key = x_api_key or DEEPSEEK_CONFIG["api_key"]
        if not api_key:
            raise HTTPException(status_code=400, detail="请配置DeepSeek API密钥")

        # 构建提示词
        prompt = f"""作为专业病理医师，分析以下医学影像检测结果：
- 图片名称: {data.image_name}
- 检测目标: {[r['class_name'] for r in data.results]}

请按以下结构回答：
1. 异常区域性质：详细描述每个检测目标的可能性质
2. 综合诊断建议：基于结果给出诊断意见
3. 进一步检查方案：推荐后续检查项目
4. 注意事项：提醒患者生活习惯和紧急情况处理

要求：专业简洁，段落分明，不使用任何符号标记。"""

        # 调用DeepSeek API
        response = requests.post(
            DEEPSEEK_CONFIG["api_url"],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": DEEPSEEK_CONFIG["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            },
            timeout=30
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"API调用失败：{response.text}")

        # 处理分析结果
        analysis = response.json()["choices"][0]["message"]["content"]
        # 使用re.sub进行正则替换
        analysis = re.sub(r'[ #*-]|^\d+\.', "", analysis, flags=re.MULTILINE)
        analysis = analysis.replace("\n\n", "\n").replace("\n", "<br><br>").strip()

        # 保存历史记录
        history_id = save_history({
            "image_name": data.image_name,
            "detection_results": data.results,
            "analysis": analysis,
            "annotated_image": data.annotated_image,
            "total_objects": data.total_objects,
            "max_confidence": data.max_confidence,
            "detect_time_ms": data.detect_time_ms
        }, current_user.username)

        return {
            "status": "success",
            "data": {
                "analysis": analysis,
                "history_id": history_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败：{str(e)}")


# 6. 获取历史记录列表
@app.get("/api/history")
async def get_history(current_user: User = Depends(get_current_active_user)):
    """获取当前用户的分析历史记录"""
    history_list = []
    user_history_dir = os.path.join(HISTORY_DIR, current_user.username)
    if os.path.exists(user_history_dir):
        for filename in os.listdir(user_history_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(user_history_dir, filename), "r", encoding="utf-8") as f:
                        history_list.append(json.load(f))
                except Exception as e:
                    print(f"读取历史记录失败：{filename} - {e}")
    return {
        "status": "success",
        "data": sorted(history_list, key=lambda x: x["timestamp"], reverse=True)
    }


# 7. 获取历史记录详情
@app.get("/api/history/{record_id}")
async def get_history_detail(record_id: str, current_user: User = Depends(get_current_active_user)):
    """获取指定ID的历史分析详情"""
    user_history_dir = os.path.join(HISTORY_DIR, current_user.username)
    if not os.path.exists(user_history_dir):
        raise HTTPException(status_code=404, detail="无历史记录")

    # 查找对应记录
    history_file = None
    for filename in os.listdir(user_history_dir):
        if filename.startswith(record_id) and filename.endswith(".json"):
            history_file = filename
            break

    if not history_file:
        raise HTTPException(status_code=404, detail=f"未找到ID为{record_id}的记录")

    with open(os.path.join(user_history_dir, history_file), "r", encoding="utf-8") as f:
        return {"status": "success", "data": json.load(f)}


# 8. 删除历史记录
@app.delete("/api/history/{record_id}")
async def delete_history(record_id: str, current_user: User = Depends(get_current_active_user)):
    """删除指定ID的历史分析记录"""
    user_history_dir = os.path.join(HISTORY_DIR, current_user.username)
    if not os.path.exists(user_history_dir):
        raise HTTPException(status_code=404, detail="无历史记录")

    # 查找对应记录
    history_file = None
    for filename in os.listdir(user_history_dir):
        if filename.startswith(record_id) and filename.endswith(".json"):
            history_file = filename
            break

    if not history_file:
        raise HTTPException(status_code=404, detail=f"未找到ID为{record_id}的记录")

    os.remove(os.path.join(user_history_dir, history_file))
    return {"status": "success", "detail": "记录已成功删除"}


# 启动服务（直接运行文件时生效）
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)