import json
import base64
import cv2
import numpy as np
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import config
from face_recognition import FaceRecognitionSystem

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

# Khởi tạo hệ thống nhận diện khuôn mặt với model Facenet
try:
    face_system = FaceRecognitionSystem(model_name="Facenet", threshold=0.6)
    logger.info("✅ Khởi tạo hệ thống nhận diện khuôn mặt thành công")
except Exception as e:
    logger.error(f"❌ Lỗi khởi tạo hệ thống nhận diện khuôn mặt: {e}")
    raise

# Khởi tạo kết nối MongoDB
try:
    client = AsyncIOMotorClient(config.MONGO_URI)
    db = client["IoT"]
    faces_collection = db["face_embeddings"]
    logger.info("✅ Kết nối MongoDB thành công")
except Exception as e:
    logger.error(f"❌ Lỗi kết nối MongoDB: {e}")
    raise

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái hoạt động của hệ thống."""
    try:
        await db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
            "face_system": "running",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Hệ thống không khả dụng")

def validate_image(image_data: str) -> bool:
    """Kiểm tra tính hợp lệ của dữ liệu ảnh."""
    if not image_data:
        return False
    try:
        # Kiểm tra định dạng base64
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

        # Kiểm tra xem userID đã tồn tại chưa
        existing_face = await faces_collection.find_one({"userID": user_id})
        if existing_face:
            raise ValueError("UserID đã được đăng ký")

        embedding = face_system.get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        # Tạo document với timestamp
        face_document = {
            "userID": user_id,
            "embeddings": embedding,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }

        await faces_collection.insert_one(face_document)
        
        await websocket.send_text(json.dumps({
            "status": "success",
            "message": "Đăng ký khuôn mặt thành công",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))

    except ValueError as e:
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": str(e)
        }))
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình đăng ký khuôn mặt: {e}")
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": "Đăng ký khuôn mặt thất bại"
        }))

async def handle_recognize_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Xử lý yêu cầu nhận diện khuôn mặt."""
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")

        new_embedding = face_system.get_embedding(image)
        if new_embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        stored_face = await faces_collection.find_one({
            "userID": user_id,
            "embeddings": {"$exists": True}
        })

        if not stored_face:
            raise ValueError("Không tìm thấy dữ liệu khuôn mặt đã đăng ký")

        if face_system.compare_faces(stored_face["embeddings"], new_embedding):
            # Cập nhật thời gian nhận dạng gần nhất
            await faces_collection.update_one(
                {"userID": user_id},
                {"$set": {"last_recognized_at": datetime.now(timezone.utc)}}
            )
            
            await websocket.send_text(json.dumps({
                "status": "success",
                "message": f"Nhận dạng khuôn mặt thành công: {user_id}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
        else:
            raise ValueError("Khuôn mặt không khớp")

    except ValueError as e:
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": str(e)
        }))
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình nhận dạng khuôn mặt: {e}")
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": "Nhận dạng khuôn mặt thất bại"
        }))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Xử lý kết nối và tin nhắn WebSocket."""
    await websocket.accept()
    logger.info(f"🔗 Kết nối WebSocket mới từ: {websocket.client}")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            # Kiểm tra loại yêu cầu
            if "type" not in data:
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Thiếu loại yêu cầu"
                }))
                continue

            # Kiểm tra và xử lý dữ liệu ảnh
            image_data = data.get("image")
            if not validate_image(image_data):
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Dữ liệu ảnh không hợp lệ hoặc thiếu"
                }))
                continue

            image = decode_image(image_data)
            if image is None:
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Không thể xử lý ảnh"
                }))
                continue

            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Loại yêu cầu không hợp lệ"
                }))

    except WebSocketDisconnect:
        logger.info(f"🔌 Đóng kết nối từ: {websocket.client}")
    except Exception as e:
        logger.error(f"❌ Lỗi không mong muốn: {e}")
    finally:
        logger.info(f"🔌 Đóng kết nối từ: {websocket.client}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)