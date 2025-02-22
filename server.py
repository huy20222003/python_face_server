import json
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
import base64
from bson import ObjectId

# Giữ các thiết lập môi trường ban đầu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
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
        face_system = FaceRecognitionSystem(
            model_path="models/arcface_model.tflite", threshold=0.8)
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

# Các endpoint API cơ bản


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


def process_binary_image(binary_data: bytes) -> np.ndarray:
    """Xử lý dữ liệu ảnh nhị phân thành numpy array."""
    try:
        np_arr = np.frombuffer(binary_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý ảnh nhị phân: {e}")
        return None

# Vẫn giữ hàm decode_image để hỗ trợ tương thích ngược (nếu cần)


def decode_image(image_data: str) -> np.ndarray:
    """Giải mã dữ liệu ảnh từ base64 sang numpy array."""
    try:
        img = base64.b64decode(image_data)
        np_arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Lỗi giải mã ảnh: {e}")
        return None

# Lưu trữ thông tin phiên WebSocket


class WebSocketSession:
    def __init__(self):
        self.current_request = None
        self.user_id = None
        self.request_type = None


# Từ điển lưu trữ phiên cho mỗi kết nối WebSocket
websocket_sessions = {}


async def handle_add_face(websocket: WebSocket, user_id: str, image: np.ndarray):
    """Xử lý yêu cầu đăng ký khuôn mặt."""
    try:
        if not user_id:
            raise ValueError("Thiếu userID")

         # Convert string userID to ObjectId
        try:
            user_id_object = ObjectId(user_id)
        except Exception:
            raise ValueError("UserID không hợp lệ")


        logger.info(f"👤 Đăng ký khuôn mặt mới cho userID: {user_id}")

        # Lấy danh sách các embedding từ ảnh
        embeddings = get_face_system().get_embeddings(image)
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Không thể tạo embedding từ khuôn mặt")

        # Với mỗi embedding được phát hiện, tạo một document riêng
        for emb in embeddings:
            face_document = {
                "userID": user_id_object,
                "name": "unknown",
                "embedding": emb.tolist(),
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            await faces_collection.insert_one(face_document)

        logger.info("✅ Đăng ký khuôn mặt thành công")
        await websocket.send_text(json.dumps({
            "type": "addFace",
            "messageFlow": "response",
            "status": "success",
            "message": "Đăng ký khuôn mặt thành công",
            "userID": user_id
        }, ensure_ascii=False))

    except ValueError as e:
        await websocket.send_text(json.dumps({
            "type": "addFace",
            "messageFlow": "response",
            "status": "fail",
            "message": str(e),
            "userID": user_id
        }, ensure_ascii=False))
    except Exception as e:
        logger.error(f"❌ Lỗi đăng ký khuôn mặt: {e}")
        await websocket.send_text(json.dumps({
            "type": "addFace",
            "messageFlow": "response",
            "status": "fail",
            "message": "Đăng ký khuôn mặt thất bại",
            "userID": user_id
        }, ensure_ascii=False))
    finally:
        gc.collect()


async def handle_recognize_face(websocket: WebSocket, user_id: str, image: np.ndarray):
    """Xử lý yêu cầu nhận diện khuôn mặt."""
    try:
        if not user_id:
            raise ValueError("Thiếu userID")
        
        logger.info("🔍 Bắt đầu nhận diện nhiều khuôn mặt trong ảnh...")

        # Giả sử hệ thống có thể phát hiện nhiều khuôn mặt trong ảnh
        face_embeddings = get_face_system().get_embeddings(image)
        if not face_embeddings:
            raise ValueError("Không tìm thấy khuôn mặt nào trong ảnh")

        # Lấy tất cả các khuôn mặt từ MongoDB
        faces = await faces_collection.find().to_list(length=None)
        if not faces:
            raise ValueError("Không có dữ liệu khuôn mặt trong hệ thống")

        recognized_results = []
        threshold = get_face_system().threshold
        face_id_counter = 1  # Biến đếm để gán face_id

        # Với mỗi embedding (từng khuôn mặt được phát hiện)
        for embedding in face_embeddings:
            min_distance = float("inf")
            best_match = None

            # So sánh với từng document
            for face in faces:
                face["_id"] = str(face["_id"])
                stored_embedding = np.array(face["embedding"], dtype=np.float32)
                distance = np.linalg.norm(embedding - stored_embedding)
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    best_match = face

            if best_match:
                recognized_results.append({
                    "face_id": face_id_counter,  # Thêm face_id
                    "userID": str(best_match["_id"]),
                    "name": best_match.get("name", "unknown"),
                    "distance": float(min_distance)
                })
                face_id_counter += 1  # Tăng số thứ tự cho khuôn mặt tiếp theo

        if recognized_results:
            logger.info(f"✅ Nhận diện thành công: {recognized_results}")
        else:
            logger.info("❌ Không tìm thấy khuôn mặt phù hợp")

        await websocket.send_text(json.dumps({
            "type": "recognizeFace",
            "messageFlow": "response",
            "status": "success" if recognized_results else "fail",
            "recognizedUsers": recognized_results,
            "userID": user_id
        }, ensure_ascii=False))

    except ValueError as e:
        await websocket.send_text(json.dumps({
            "type": "recognizeFace", 
            "messageFlow": "response",
            "status": "fail", 
            "message": str(e),
            "userID": user_id
        }, ensure_ascii=False))
    except Exception as e:
        logger.error(f"❌ Lỗi nhận diện khuôn mặt: {e}")
        await websocket.send_text(json.dumps({
            "type": "recognizeFace", 
            "messageFlow": "response",
            "status": "fail", 
            "message": "Nhận diện khuôn mặt thất bại",
            "userID": user_id
        }, ensure_ascii=False))
    finally:
        gc.collect()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Xử lý kết nối WebSocket với hỗ trợ nhận dữ liệu nhị phân."""
    await websocket.accept()
    client_id = id(websocket)
    websocket_sessions[client_id] = WebSocketSession()
    session = websocket_sessions[client_id]
    
    logger.info(f"🔗 Kết nối WebSocket mới từ client: {websocket.client}")
    
    try:
        while True:
            # Nhận tin nhắn từ client
            message = await websocket.receive()
            
            # Xử lý tin nhắn dựa vào loại dữ liệu
            if "text" in message:
                # Nhận metadata JSON
                data = json.loads(message["text"])
                logger.info(f"📩 Nhận metadata: {data}")
                
                request_type = data.get("type")
                if request_type in ["addFace", "recognizeFace"]:
                    # Lưu thông tin yêu cầu vào phiên
                    session.request_type = request_type
                    session.user_id = data.get("userID")
                    
                    # Nếu là tin nhắn legacy hoặc hỗn hợp có cả base64
                    if "image" in data:
                        # Xử lý phương thức cũ với base64
                        image = decode_image(data.get("image"))
                        if request_type == "addFace":
                            await handle_add_face(websocket, session.user_id, image)
                        elif request_type == "recognizeFace":
                            await handle_recognize_face(websocket, session.user_id, image)
                    else:
                        # Nếu không có ảnh, đợi tin nhắn nhị phân tiếp theo
                        logger.info(f"⏳ Đang đợi dữ liệu ảnh nhị phân cho yêu cầu {request_type}...")
                else:
                    await websocket.send_text(json.dumps({
                        "status": "fail", 
                        "messageFlow": "response",
                        "message": "Loại yêu cầu không hợp lệ"
                    }, ensure_ascii=False))
            
            elif "bytes" in message:
                # Nhận dữ liệu ảnh nhị phân
                binary_data = message["bytes"]
                logger.info(f"📸 Đã nhận {len(binary_data)} bytes dữ liệu ảnh")
                
                if session.request_type:
                    # Xử lý dữ liệu ảnh nhị phân
                    image = process_binary_image(binary_data)
                    
                    if image is not None:
                        if session.request_type == "addFace":
                            await handle_add_face(websocket, session.user_id, image)
                        elif session.request_type == "recognizeFace":
                            await handle_recognize_face(websocket, session.user_id, image)
                    else:
                        await websocket.send_text(json.dumps({
                            "type": session.request_type,
                            "messageFlow": "response",
                            "status": "fail", 
                            "message": "Không thể xử lý dữ liệu ảnh"
                        }, ensure_ascii=False))
                else:
                    await websocket.send_text(json.dumps({
                        "status": "fail", 
                        "messageFlow": "response",
                        "message": "Nhận dữ liệu ảnh trước khi có yêu cầu"
                    }, ensure_ascii=False))
                
                # Reset thông tin phiên sau khi xử lý xong
                session.request_type = None
                session.user_id = None

    except WebSocketDisconnect:
        logger.info(f"🔌 Đóng kết nối WebSocket từ client: {websocket.client}")
    except Exception as e:
        logger.error(f"❌ Lỗi WebSocket: {e}")
    finally:
        # Dọn dẹp phiên khi kết nối đóng
        if client_id in websocket_sessions:
            del websocket_sessions[client_id]
        logger.info(f"🧹 Đã dọn dẹp phiên WebSocket cho client: {websocket.client}")
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)