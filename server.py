import json
import base64
import cv2
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path
import config
from face_recognition import FaceRecognitionSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognition system
try:
    MODEL_PATH = Path("models") / "facenet_weights.h5"
    face_system = FaceRecognitionSystem(MODEL_PATH)
    logger.info("✅ Face recognition system initialized successfully")
except Exception as e:
    logger.error(f"❌ Face recognition system initialization failed: {e}")
    raise

# Initialize MongoDB connection
try:
    client = AsyncIOMotorClient(config.MONGO_URI)
    db = client["IoT"]
    faces_collection = db["face_embeddings"]
    logger.info("✅ MongoDB connection successful")
except Exception as e:
    logger.error(f"❌ MongoDB connection error: {e}")
    raise

@app.get("/")
async def health_check():
    return {"status": "running"}

@app.head("/")
async def reject_head():
    return {}

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    try:
        img = base64.b64decode(image_data)
        np_arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Image decoding error: {e}")
        return None

async def handle_add_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Handle face registration request."""
    try:
        user_id = data.get("userID")
        if not user_id:
            raise ValueError("Thiếu userID")

        embedding = face_system.get_embedding(image)
        if embedding is None:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        await faces_collection.insert_one({
            "userID": user_id,
            "embeddings": embedding
        })
        
        await websocket.send_text(json.dumps({
            "status": "success",
            "message": "Đăng ký khuôn mặt thành công"
        }))

    except ValueError as e:
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": str(e)
        }))
    except Exception as e:
        logger.error(f"❌ Error in handle_add_face: {e}")
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": "Đăng ký khuôn mặt thất bại"
        }))

async def handle_recognize_face(websocket: WebSocket, data: dict, image: np.ndarray):
    """Handle face recognition request."""
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

        if stored_face and face_system.compare_faces(
            stored_face["embeddings"],
            new_embedding
        ):
            await websocket.send_text(json.dumps({
                "status": "success",
                "message": f"Nhận dạng khuôn mặt thành công: {user_id}"
            }))
        else:
            raise ValueError("Face not recognized")

    except ValueError as e:
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": str(e)
        }))
    except Exception as e:
        logger.error(f"❌ Error in handle_recognize_face: {e}")
        await websocket.send_text(json.dumps({
            "status": "fail",
            "message": "Nhận dạng khuôn mặt thất bại"
        }))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections and messages."""
    await websocket.accept()
    logger.info(f"🔗 New WebSocket connection from: {websocket.client}")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            image_data = data.get("image")
            if not image_data:
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Missing image data"
                }))
                continue

            image = decode_image(image_data)
            if image is None:
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Invalid image data"
                }))
                continue

            if data["type"] == "addFace":
                await handle_add_face(websocket, data, image)
            elif data["type"] == "recognizeFace":
                await handle_recognize_face(websocket, data, image)
            else:
                await websocket.send_text(json.dumps({
                    "status": "fail",
                    "message": "Invalid request type"
                }))

    except WebSocketDisconnect:
        logger.info(f"🔌 Connection closed from: {websocket.client}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        logger.info(f"🔌 Connection closed from: {websocket.client}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)