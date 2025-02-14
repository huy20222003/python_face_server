import asyncio
import json
import base64
import cv2
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from face_recognition import get_embedding, compare_faces
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn (có thể đổi sang domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức (GET, POST, OPTIONS, ...)
    allow_headers=["*"],  # Cho phép tất cả header
)

# Kết nối MongoDB bằng motor
try:
    client = AsyncIOMotorClient(config.MONGO_URI)
    db = client["IoT"]
    faces_collection = db["face_embeddings"]
    logging.info("✅ MongoDB connection successful")
except Exception as e:
    logging.error(f"❌ MongoDB connection error: {e}")
    raise

# Route kiểm tra trạng thái server
@app.get("/")
async def health_check():
    return {"status": "running"}

@app.head("/")
async def reject_head():
    return {}

# Xử lý WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info(f"🔗 New WebSocket connection from: {websocket.client}")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            image_data = data.get("image")
            if not image_data:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Missing image data"}))
                continue

            image = decode_image(image_data)
            if image is None:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Invalid image data"}))
                continue

            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send_text(json.dumps({"status": "fail", "message": "Invalid request type"}))

    except WebSocketDisconnect:
        logging.info(f"🔌 Connection closed from: {websocket.client}")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
    finally:
        logging.info(f"🔌 Connection closed from: {websocket.client}")

# Hàm giải mã ảnh từ base64
def decode_image(image_data):
    try:
        img = base64.b64decode(image_data)
        np_arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"❌ Image decoding error: {e}")
        return None

# Xử lý đăng ký khuôn mặt
async def handle_add_face(websocket: WebSocket, data, image):
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")

        # # Kiểm tra xem userID đã tồn tại chưa
        # existing_user = await faces_collection.find_one({"userID": user_id})
        # if existing_user:
        #     raise ValueError("Người dùng này đã đăng ký khuôn mặt trước đó")

        embedding = get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        # Thêm dữ liệu vào MongoDB nếu chưa tồn tại
        await faces_collection.insert_one({"userID": user_id, "embeddings": embedding})
        await websocket.send_text(json.dumps({"status": "success", "message": "Đăng ký khuôn mặt thành công"}))

    except ValueError as e:
        await websocket.send_text(json.dumps({"status": "fail", "message": str(e)}))
    except Exception as e:
        logging.error(f"❌ Lỗi trong handle_add_face: {e}")
        await websocket.send_text(json.dumps({"status": "fail", "message": "Đăng ký khuôn mặt thất bại"}))


# Xử lý nhận diện khuôn mặt
async def handle_recognize_face(websocket: WebSocket, data, image):
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")

        embedding = get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        match = await faces_collection.find_one({"userID": user_id, "embeddings": {"$exists": True}})
        if match and compare_faces(match["embeddings"], embedding):
            await websocket.send_text(json.dumps({"status": "success", "message": f"Nhận dạng khuôn mặt thành công: {user_id}"}))
        else:
            raise ValueError("Face not recognized")
    except ValueError as e:
        await websocket.send_text(json.dumps({"status": "fail", "message": str(e)}))
    except Exception as e:
        logging.error(f"❌ Error in handle_recognize_face: {e}")
        await websocket.send_text(json.dumps({"status": "fail", "message": "Nhận dạng khuôn mặt thất bại"}))

# Chạy server FastAPI trên cổng 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
