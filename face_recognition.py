from deepface import DeepFace
import numpy as np

# Hàm tạo embedding từ ảnh
def get_embedding(image):
    try:
        embedding = DeepFace.represent(image, model_name="Facenet")[0]["embedding"]
        return embedding
    except:
        return None

# Hàm so sánh khuôn mặt
def compare_faces(embedding1, embedding2, threshold=0.7):
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    return distance < threshold
