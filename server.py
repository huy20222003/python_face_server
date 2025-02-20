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
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
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
        face_system = FaceRecognitionSystem(threshold=0.8)
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

@app.get("/")
async def root():
    return {"message": "Welcome to the Face Recognition API!"}

@app.get("/health")
async def health_check():
    try:
        await db.command("ping")
        return {"status": "healthy", "database": "connected", "face_system": "running"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Hệ thống không khả dụng")

def decode_image(image_data: str) -> np.ndarray:
    try:
        img = base64.b64decode(image_data)
        np_arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Lỗi giải mã ảnh: {e}")
        return None

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    return embedding / np.linalg.norm(embedding)

async def handle_add_face(websocket: WebSocket, data: dict, image: np.ndarray):
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")

        embeddings = get_face_system().get_embeddings(image)
        if not embeddings:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        for emb in embeddings:
            normalized_embedding = normalize_embedding(emb)
            face_document = {
                "userID": user_id,
                "name": "unknown",
                "embedding": normalized_embedding.tolist(),
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            await faces_collection.insert_one(face_document)

        await websocket.send_text(json.dumps({"type": "addFace", "status": "success"}, ensure_ascii=False))
    except Exception as e:
        logger.error(f"❌ Lỗi đăng ký khuôn mặt: {e}")
        await websocket.send_text(json.dumps({"type": "addFace", "status": "fail", "message": str(e)}, ensure_ascii=False))
    finally:
        gc.collect()

async def handle_recognize_face(websocket: WebSocket, data: dict, image: np.ndarray):
    try:
        face_embeddings = get_face_system().get_embeddings(image)
        if not face_embeddings:
            raise ValueError("Không tìm thấy khuôn mặt nào trong ảnh")

        faces = await faces_collection.find().to_list(length=None)
        if not faces:
            raise ValueError("Không có dữ liệu khuôn mặt trong hệ thống")

        recognized_results = []
        threshold = get_face_system().threshold

        for embedding in face_embeddings:
            embedding = normalize_embedding(embedding)
            best_match = None
            max_similarity = -1

            for face in faces:
                stored_embedding = np.array(face["embedding"], dtype=np.float32)
                similarity = np.dot(embedding, stored_embedding)
                
                if similarity > max_similarity and similarity > threshold:
                    max_similarity = similarity
                    best_match = face

            if best_match:
                recognized_results.append({
                    "userID": best_match["userID"],
                    "name": best_match.get("name", "unknown"),
                    "similarity": float(max_similarity)
                })

        await websocket.send_text(json.dumps({
            "type": "recognizeFace", "status": "success" if recognized_results else "fail",
            "recognizedUsers": recognized_results
        }, ensure_ascii=False))
    except Exception as e:
        logger.error(f"❌ Lỗi nhận diện khuôn mặt: {e}")
        await websocket.send_text(json.dumps({"type": "recognizeFace", "status": "fail", "message": str(e)}, ensure_ascii=False))
    finally:
        gc.collect()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            image = decode_image(data.get("image"))
            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Loại yêu cầu không hợp lệ"}, ensure_ascii=False))
    except WebSocketDisconnect:
        logger.info("🔌 Đóng kết nối WebSocket")
    finally:
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
