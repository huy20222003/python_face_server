import json
import base64
import cv2
import numpy as np
import logging
import gc
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import config
import tensorflow as tf
import os
from face_recognition import FaceRecognitionSystem

# Tắt log không cần thiết của TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Thiết lập logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Face Recognition API")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

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
        face_system = FaceRecognitionSystem(model_name="ArcFace", threshold=0.6)
        logger.info("✅ Hệ thống nhận diện khuôn mặt đã được khởi tạo")
    return face_system

# Kết nối MongoDB
try:
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
    """Kiểm tra trạng thái hoạt động của hệ thống."""
    try:
        await db.command("ping")
        return {"status": "healthy", "database": "connected", "face_system": "running", "timestamp": datetime.now(timezone.utc).isoformat()}
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

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize ảnh về kích thước nhỏ hơn để giảm tải xử lý"""
    target_size = (160, 160)
    return cv2.resize(image, target_size)

async def handle_add_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Xử lý yêu cầu đăng ký khuôn mặt."""
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")

        existing_face = await faces_collection.find_one({"userID": user_id})
        if existing_face:
            raise ValueError("UserID đã được đăng ký")

        image = preprocess_image(image)
        embedding = get_face_system().get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        face_document = {"userID": user_id, "embeddings": np.array(embedding, dtype=np.float32).tolist(), "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)}

        await faces_collection.insert_one(face_document)
        
        await websocket.send_text(json.dumps({ "type": "addFace", "status": "success", "message": "Đăng ký khuôn mặt thành công", "messageFlow": "response" }))

    except ValueError as e:
        await websocket.send_text(json.dumps({"type": "addFace", "status": "fail", "message": str(e), "messageFlow": "response"}))
    except Exception as e:
        logger.error(f"❌ Lỗi đăng ký khuôn mặt: {e}")
        await websocket.send_text(json.dumps({"type": "addFace", "status": "fail", "message": "Đăng ký khuôn mặt thất bại", "messageFlow": "response"}))
    finally:
        gc.collect()
        
async def handle_recognize_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Xử lý yêu cầu nhận dạng khuôn mặt."""
    try:
        image = preprocess_image(image)
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
            distance = np.linalg.norm(embedding - stored_embedding)  # Tính khoảng cách Euclidean
            
            if distance < get_face_system().threshold and distance < min_distance:
                min_distance = distance
                recognized_user = face["userID"]

        if recognized_user:
            response = {
                "type": "recognizeFace",
                "status": "success",
                "message": "Nhận diện thành công",
                "userID": recognized_user,
                "distance": min_distance,
                "messageFlow": "response",
            }
        else:
            response = {
                "type": "recognizeFace", 
                "status": "fail",
                "message": "Khuôn mặt chưa được đăng ký",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "messageFlow": "response"
            }

        await websocket.send_text(json.dumps(response))

    except ValueError as e:
        await websocket.send_text(json.dumps({"type": "recognizeFace", "status": "fail", "message": str(e), "messageFlow": "response"}))
    except Exception as e:
        logger.error(f"❌ Lỗi nhận diện khuôn mặt: {e}")
        await websocket.send_text(json.dumps({"type": "recognizeFace", "status": "fail", "message": "Nhận diện khuôn mặt thất bại", "messageFlow": "response"}))
    finally:
        gc.collect()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Xử lý kết nối WebSocket."""
    await websocket.accept()
    logger.info(f"🔗 Kết nối WebSocket mới từ: {websocket.client}")
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if "type" not in data:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Thiếu loại yêu cầu"}))
                continue

            image_data = data.get("image")
            if not validate_image(image_data):
                await websocket.send_text(json.dumps({"status": "fail", "message": "Dữ liệu ảnh không hợp lệ hoặc thiếu"}))
                continue

            image = decode_image(image_data)
            if image is None:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Không thể xử lý ảnh"}))
                continue

            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Loại yêu cầu không hợp lệ"}))

    except WebSocketDisconnect:
        logger.info(f"🔌 Đóng kết nối từ: {websocket.client}")
    except Exception as e:
        logger.error(f"❌ Lỗi không mong muốn: {e}")
    finally:
        logger.info(f"🔌 Đóng kết nối từ: {websocket.client}")
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)