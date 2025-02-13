from deepface import DeepFace
import numpy as np
import logging

# Cấu hình logging để debug dễ hơn
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hàm tạo embedding từ ảnh
def get_embedding(image):
    try:
        result = DeepFace.represent(image, model_name="Facenet")
        if result and isinstance(result, list) and "embedding" in result[0]:
            return result[0]["embedding"]
        else:
            logging.error("❌ Không tìm thấy embedding trong kết quả từ DeepFace.")
            return None
    except Exception as e:
        logging.error(f"❌ Lỗi trong quá trình tạo embedding: {e}")
        return None

# Hàm so sánh khuôn mặt
def compare_faces(embedding1, embedding2, threshold=0.5):
    try:
        if embedding1 is None or embedding2 is None:
            logging.warning("⚠️ Một trong hai embedding bị None, không thể so sánh.")
            return False

        distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
        return distance < threshold
    except Exception as e:
        logging.error(f"❌ Lỗi trong quá trình so sánh khuôn mặt: {e}")
        return False
