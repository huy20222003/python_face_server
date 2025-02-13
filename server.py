import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI
from face_recognition import get_embedding, compare_faces
from pymongo import MongoClient
import config
import logging
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Route kiểm tra trạng thái server
@app.get("/")
async def health_check():
    return {"status": "running"}

@app.head("/")
async def reject_head():
    return {}

class WebSocketServer:
    def __init__(self):
        try:
            self.client = MongoClient(config.MONGO_URI)
            self.db = self.client["iot"]
            self.faces_collection = self.db["face_embeddings"]
            logging.info("✅ MongoDB connection successful")
        except Exception as e:
            logging.error(f"❌ MongoDB connection error: {e}")
            raise

    async def handle_connection(self, websocket, path):
        try:
            logging.info(f"🔗 New WebSocket connection from: {websocket.remote_address}")
            async for message in websocket:
                await self.process_message(websocket, message)
        except ConnectionClosed:
            logging.info(f"🔌 Connection closed from: {websocket.remote_address}")
        except Exception as e:
            logging.error(f"❌ Unexpected error: {e}")
        finally:
            logging.info(f"🔌 Connection closed from: {websocket.remote_address}")

    async def process_message(self, websocket, message):
        try:
            data = json.loads(message)
            if not isinstance(data, dict):
                raise ValueError("Invalid message format")

            image_data = data.get("image")
            if not image_data:
                raise ValueError("Missing image data")

            image = self.decode_image(image_data)
            if image is None:
                raise ValueError("Invalid image data")

            if data["type"] == "addFace":
                await self.handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await self.handle_recognize_face(websocket, data, image)
            else:
                raise ValueError("Invalid request type")
        except (json.JSONDecodeError, ValueError) as e:
            await websocket.send(json.dumps({"status": "fail", "message": str(e)}))

    def decode_image(self, image_data):
        try:
            img = base64.b64decode(image_data)
            np_arr = np.frombuffer(img, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"❌ Image decoding error: {e}")
            return None

    async def handle_add_face(self, websocket, data, image):
        try:
            user_id = data.get("userID")
            if not user_id:
                raise ValueError("Missing userID")

            embedding = get_embedding(image)
            if embedding is None:
                raise ValueError("Could not generate face embedding")

            await self.faces_collection.insert_one({"embeddings": embedding, "userID": user_id})
            await websocket.send(json.dumps({"status": "success", "message": "Face registration successful"}))
        except ValueError as e:
            await websocket.send(json.dumps({"status": "fail", "message": str(e)}))
        except Exception as e:
            logging.error(f"❌ Error in handle_add_face: {e}")
            await websocket.send(json.dumps({"status": "fail", "message": "Face registration failed"}))

    async def handle_recognize_face(self, websocket, data, image):
        try:
            user_id = data.get("userID")
            if not user_id:
                raise ValueError("Missing userID")

            embedding = get_embedding(image)
            if embedding is None:
                raise ValueError("Could not generate face embedding")

            match = await self.faces_collection.find_one({"userID": user_id, "embeddings": {"$exists": True}})
            if match and compare_faces(match["embeddings"], embedding):
                await websocket.send(json.dumps({"status": "success", "message": f"Face recognition successful: {user_id}"}))
            else:
                raise ValueError("Face not recognized")
        except ValueError as e:
            await websocket.send(json.dumps({"status": "fail", "message": str(e)}))
        except Exception as e:
            logging.error(f"❌ Error in handle_recognize_face: {e}")
            await websocket.send(json.dumps({"status": "fail", "message": "Face recognition failed"}))

async def websocket_server():
    server = WebSocketServer()
    async with websockets.serve(server.handle_connection, "0.0.0.0", 5000, ping_interval=30, ping_timeout=10):
        logging.info("🚀 WebSocket server running on port 5000")
        await asyncio.Future()  # Chạy mãi mãi

if __name__ == "__main__":
    import uvicorn
    loop = asyncio.get_event_loop()
    loop.create_task(websocket_server())
    uvicorn.run(app, host="0.0.0.0", port=8000)
