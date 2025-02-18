import json
import base64
import cv2
import numpy as np
import logging
import gc
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import config
import tensorflow as tf
import os
from face_recognition import FaceRecognitionSystem

# Tắt log không cần thiết của TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Tắt tối ưu hóa CPU
os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"  # Vô hiệu hóa XNNPACK

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,  # Đặt mức log để debug
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Face Recognition API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy Loading hệ thống nhận diện khuôn mặt
face_system = None

def get_face_system():
    global face_system
    if face_system is None:
        logger.info("🟢 Khởi tạo hệ thống nhận diện khuôn mặt...")
        face_system = FaceRecognitionSystem(model_path="models/arcface_model.tflite", threshold=0.6)
        logger.info("✅ Hệ thống nhận diện khuôn mặt đã sẵn sàng")
    return face_system

# Kết nối MongoDB
try:
    logger.info("🔄 Kết nối đến MongoDB...")
    client = AsyncIOMotorClient(config.MONGO_URI)
    db = client["IoT"]
    faces_collection = db["face_embeddings"]
    logger.info("✅ Kết nối MongoDB thành công")
except Exception as e:
    logger.error(f"❌ Lỗi kết nối MongoDB: {e}")
    raise

#Route kiểm tra trạng thái server
@app.head("/")
async def reject_head():
    return {}

@app.get("/")
async def root():
    return {"message": "Welcome to the Face Recognition API!"}

@app.get("/health")
async def health_check():
    try:
        logger.info("🔍 Kiểm tra trạng thái hệ thống...")
        await db.command("ping")
        return {"status": "healthy", "database": "connected", "face_system": "running"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Hệ thống không khả dụng")

def validate_image(image_data: str) -> bool:
    """Kiểm tra tính hợp lệ của dữ liệu ảnh."""
    if not image_data:
        return False
    try:
        base64.b64decode(image_data)
        return True
    except Exception:
        return False

def decode_image(image_data: str) -> np.ndarray:
    """Giải mã dữ liệu ảnh từ base64 sang numpy array."""
    try:
        img = base64.b64decode(image_data)
        np_arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Lỗi giải mã ảnh: {e}")
        return None

async def handle_add_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Xử lý yêu cầu đăng ký khuôn mặt."""
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")
        logger.info(f"👤 Đăng ký khuôn mặt mới cho userID: {user_id}")
        existing_face = await faces_collection.find_one({"userID": user_id})
        if existing_face:
            raise ValueError("UserID đã được đăng ký")

        embedding = get_face_system().get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")
        face_document = {"userID": user_id, "embeddings": embedding.tolist(), "created_at": datetime.now(timezone.utc)}
        await faces_collection.insert_one(face_document)
        logger.info("✅ Đăng ký khuôn mặt thành công")
        await websocket.send_text(json.dumps({"type": "addFace", "status": "success"}))
    except ValueError as e:
        await websocket.send_text(json.dumps({"type": "addFace", "status": "fail", "message": str(e)}))
    except Exception as e:
        logger.error(f"❌ Lỗi đăng ký khuôn mặt: {e}")
        await websocket.send_text(json.dumps({"type": "addFace", "status": "fail", "message": "Đăng ký khuôn mặt thất bại", "messageFlow": "response"}))
    finally:
        gc.collect()

async def handle_recognize_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Xử lý yêu cầu nhận dạng khuôn mặt."""
    try:
        logger.info("🔍 Bắt đầu nhận diện khuôn mặt...")
        embedding = get_face_system().get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        # Lấy tất cả embeddings từ MongoDB
        faces = await faces_collection.find().to_list(length=None)
        if not faces:
            raise ValueError("Không có dữ liệu khuôn mặt trong hệ thống")

        min_distance = float("inf")
        recognized_user = None

        for face in faces:
            stored_embedding = np.array(face["embeddings"], dtype=np.float32)
            distance = np.linalg.norm(embedding - stored_embedding)
            if distance < get_face_system().threshold and distance < min_distance:
                min_distance = distance
                recognized_user = face["userID"]

        if recognized_user:
            logger.info(f"✅ Nhận diện thành công: {recognized_user}")
        else:
            logger.info("❌ Không tìm thấy khuôn mặt phù hợp")
        await websocket.send_text(json.dumps({"type": "recognizeFace", "status": "success" if recognized_user else "fail", "userID": recognized_user}))
    except ValueError as e:
        await websocket.send_text(json.dumps({"type": "recognizeFace", "status": "fail", "message": str(e)}))
    except Exception as e:
        logger.error(f"❌ Lỗi nhận diện khuôn mặt: {e}")
        await websocket.send_text(json.dumps({"type": "recognizeFace", "status": "fail", "message": "Nhận diện khuôn mặt thất bại", "messageFlow": "response"}))
    finally:
        gc.collect()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Xử lý kết nối WebSocket."""
    await websocket.accept()
    logger.info("🔗 Kết nối WebSocket mới")
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            logger.info(f"📩 Nhận dữ liệu WebSocket: {data}")
            image = decode_image(data.get("image"))
            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Loại yêu cầu không hợp lệ"}))

    except WebSocketDisconnect:
        logger.info("🔌 Đóng kết nối WebSocket")
    except Exception as e:
        logger.error(f"❌ Lỗi WebSocket: {e}")
    finally:
        logger.info(f"🔌 Đóng kết nối từ: {websocket.client}")
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)