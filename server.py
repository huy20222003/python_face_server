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
from websockets.exceptions import InvalidHandshake, ConnectionClosed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

    async def validate_websocket_request(self, websocket):
        if not websocket.request_headers.get("Upgrade", "").lower() == "websocket":
            logging.warning(f"Invalid connection attempt from {websocket.remote_address}")
            return False
        
        if websocket.request_headers.get("Connection", "").lower() != "upgrade":
            logging.warning(f"Missing or invalid Connection header from {websocket.remote_address}")
            return False
            
        return True

    async def handle_connection(self, websocket, path):
        if not await self.validate_websocket_request(websocket):
            return

        logging.info(f"🔗 New WebSocket connection from: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if not isinstance(data, dict):
                        await websocket.send(json.dumps({
                            "status": "fail",
                            "message": "Invalid message format"
                        }))
                        continue

                    image_data = data.get("image")
                    if not image_data:
                        await websocket.send(json.dumps({
                            "status": "fail",
                            "message": "Missing image data"
                        }))
                        continue

                    image = self.decode_image(image_data)
                    if image is None:
                        await websocket.send(json.dumps({
                            "status": "fail",
                            "message": "Invalid image data"
                        }))
                        continue

                    if data["type"] == "addFace":
                        await self.handle_add_face(websocket, data, image)
                    elif data["type"] == "recognizeFace":
                        await self.handle_recognize_face(websocket, data, image)
                    else:
                        await websocket.send(json.dumps({
                            "status": "fail",
                            "message": "Invalid request type"
                        }))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "status": "fail",
                        "message": "Invalid JSON format"
                    }))
                    
        except ConnectionClosed:
            logging.info(f"Connection closed normally from {websocket.remote_address}")
        except Exception as e:
            logging.error(f"Unexpected error in connection handler: {e}")
        finally:
            logging.info(f"🔌 Connection closed from: {websocket.remote_address}")

    def decode_image(self, image_data):
        try:
            img = base64.b64decode(image_data)
            np_arr = np.frombuffer(img, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Image decoding error: {e}")
            return None

    async def handle_add_face(self, websocket, data, image):
        try:
            user_id = data.get("userID")
            if not user_id:
                await websocket.send(json.dumps({
                    "status": "fail",
                    "message": "Missing userID"
                }))
                return

            embedding = get_embedding(image)
            if embedding is None:
                await websocket.send(json.dumps({
                    "status": "fail",
                    "message": "Could not generate face embedding"
                }))
                return

            await self.faces_collection.insert_one({
                "embeddings": embedding,
                "userID": user_id
            })
            
            await websocket.send(json.dumps({
                "status": "success",
                "message": "Face registration successful"
            }))

        except Exception as e:
            logging.error(f"Error in handle_add_face: {e}")
            await websocket.send(json.dumps({
                "status": "fail",
                "message": "Face registration failed"
            }))

    async def handle_recognize_face(self, websocket, data, image):
        try:
            user_id = data.get("userID")
            if not user_id:
                await websocket.send(json.dumps({
                    "status": "fail",
                    "message": "Missing userID"
                }))
                return

            embedding = get_embedding(image)
            if embedding is None:
                await websocket.send(json.dumps({
                    "status": "fail",
                    "message": "Could not generate face embedding"
                }))
                return

            match = await self.faces_collection.find_one({
                "userID": user_id,
                "embeddings": {"$exists": True}
            })

            if match and compare_faces(match["embeddings"], embedding):
                await websocket.send(json.dumps({
                    "status": "success",
                    "message": f"Face recognition successful: {user_id}"
                }))
            else:
                await websocket.send(json.dumps({
                    "status": "fail",
                    "message": "Face not recognized"
                }))

        except Exception as e:
            logging.error(f"Error in handle_recognize_face: {e}")
            await websocket.send(json.dumps({
                "status": "fail",
                "message": "Face recognition failed"
            }))

async def main():
    server = WebSocketServer()
    try:
        async with websockets.serve(
            server.handle_connection,
            "0.0.0.0",
            5000,
            ping_interval=30,
            ping_timeout=10
        ) as websocket_server:
            logging.info("🚀 WebSocket server running on port 5000")
            await asyncio.Future()  # run forever
    except Exception as e:
        logging.error(f"Failed to start WebSocket server: {e}")

if __name__ == "__main__":
    asyncio.run(main())