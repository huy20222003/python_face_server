import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from face_recognition import get_embedding, compare_faces
from pymongo import MongoClient
import config
import logging

# Cấu hình logging để debug dễ hơn
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Kết nối MongoDB
try:
    client = MongoClient(config.MONGO_URI)
    db = client["iot"]
    faces_collection = db["face_embeddings"]
    logging.info("✅ Kết nối MongoDB thành công.")
except Exception as e:
    logging.error(f"❌ Lỗi kết nối MongoDB: {e}")
    exit(1)

# Giải mã ảnh từ base64
def decode_image(image_data):
    try:
        img = base64.b64decode(image_data)
        np_arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"❌ Lỗi giải mã ảnh: {e}")
        return None

# Xử lý kết nối WebSocket
async def handle_connection(websocket, path):
    logging.info(f"🔗 Kết nối WebSocket từ: {websocket.remote_address}")

    # Kiểm tra yêu cầu có phải WebSocket không
    if websocket.request_headers.get("Upgrade", "").lower() != "websocket":
        logging.warning("🚫 Bị từ chối: Yêu cầu không phải WebSocket")
        return

    try:
        async for message in websocket:
            logging.info(f"📥 Nhận dữ liệu từ client: {message[:100]}")  # Log giới hạn 100 ký tự đầu

            data = json.loads(message)
            image = decode_image(data.get("image", ""))

            if image is None:
                await websocket.send(json.dumps({"status": "fail", "message": "Lỗi xử lý ảnh"}))
                continue

            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send(json.dumps({"status": "fail", "message": "Loại yêu cầu không hợp lệ"}))

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("✅ Kết nối WebSocket đóng một cách an toàn.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.warning(f"⚠️ Kết nối WebSocket bị đóng đột ngột: {e}")
    except Exception as e:
        logging.error(f"❌ Lỗi không mong muốn: {e}")
    finally:
        logging.info(f"🔌 Ngắt kết nối WebSocket từ: {websocket.remote_address}")

# Xử lý thêm khuôn mặt vào database
async def handle_add_face(websocket, data, image):
    try:
        embedding = get_embedding(image)
        user_id = data.get("userID")

        if not user_id:
            await websocket.send(json.dumps({"status": "fail", "message": "Thiếu userID"}))
            return

        faces_collection.insert_one({"embeddings": embedding, "userID": user_id})
        await websocket.send(json.dumps({"status": "success", "message": "Đăng ký khuôn mặt thành công"}))

    except Exception as e:
        logging.error(f"❌ Lỗi trong handle_add_face: {e}")
        await websocket.send(json.dumps({"status": "fail", "message": "Lỗi trong quá trình đăng ký khuôn mặt"}))

# Xử lý nhận diện khuôn mặt
async def handle_recognize_face(websocket, data, image):
    try:
        embedding = get_embedding(image)
        user_id = data.get("userID")

        if not user_id:
            await websocket.send(json.dumps({"status": "fail", "message": "Thiếu userID"}))
            return

        for face in faces_collection.find({"userID": user_id}):
            if compare_faces(face["embeddings"], embedding):
                await websocket.send(json.dumps({"status": "success", "message": f"Nhận diện thành công: {user_id}"}))
                return

        await websocket.send(json.dumps({"status": "fail", "message": "Không nhận diện được khuôn mặt"}))

    except Exception as e:
        logging.error(f"❌ Lỗi trong handle_recognize_face: {e}")
        await websocket.send(json.dumps({"status": "fail", "message": "Lỗi trong quá trình nhận diện khuôn mặt"}))

# Khởi động WebSocket server
async def main():
    try:
        start_server = await websockets.serve(handle_connection, "0.0.0.0", 5000)
        logging.info("🚀 WebSocket server đang chạy trên cổng 5000...")
        await start_server.wait_closed()
    except Exception as e:
        logging.error(f"❌ Lỗi khởi động WebSocket server: {e}")

if __name__ == "__main__":
    asyncio.run(main())
