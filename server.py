import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from face_recognition import get_embedding, compare_faces
from pymongo import MongoClient
import config

# Kết nối MongoDB
client = MongoClient(config.MONGO_URI)
db = client["iot"]
faces_collection = db["face_embeddings"]

# Giải mã ảnh từ base64
def decode_image(image_data):
    img = base64.b64decode(image_data)
    np_arr = np.frombuffer(img, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# Xử lý kết nối WebSocket
async def handle_connection(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        image = decode_image(data["image"])
        
        if data["type"] == "addFace":
            embedding = get_embedding(image)
            user_id = data.get("userID")

            if user_id:
                faces_collection.insert_one({
                    "embeddings": embedding,
                    "userID": user_id
                })
                await websocket.send(json.dumps({"status": "success", "message": "Đăng ký thành công"}))
            else:
                await websocket.send(json.dumps({"status": "fail", "message": "Thiếu userID"}))

        elif data["type"] == "recognizeFace":
            embedding = get_embedding(image)
            user_id = data.get("userID")

            if user_id:
                for face in faces_collection.find({"userID": user_id}):
                    if compare_faces(face["embeddings"], embedding):
                        await websocket.send(json.dumps({
                            "status": "success",
                            "message": f"Nhận diện thành công: {face['name']}"
                        }))
                        return
                
                await websocket.send(json.dumps({"status": "fail", "message": "Không nhận diện được"}))
            else:
                await websocket.send(json.dumps({"status": "fail", "message": "Thiếu userID"}))

# Khởi động WebSocket server
async def main():
    start_server = await websockets.serve(handle_connection, "0.0.0.0", 5000)
    await start_server.wait_closed()

asyncio.run(main())  # Chạy server với asyncio.run()

