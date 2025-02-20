import numpy as np
import logging
import cv2
from typing import Optional, List
from mtcnn import MTCNN
from keras_facenet import FaceNet

def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Tính khoảng cách Euclidean giữa hai embeddings."""
    return np.linalg.norm(embedding1 - embedding2)

class FaceRecognitionSystem:
    def __init__(self, threshold: float = 0.8):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt sử dụng FaceNet.
        Args:
            threshold: Ngưỡng nhận diện (Euclidean distance).
        """
        self.threshold = threshold
        self._setup_logging()
        self.logger.info("🔄 Đang tải mô hình FaceNet...")
        self.facenet = FaceNet()
        self.logger.info("✅ Mô hình FaceNet đã tải thành công.")
        self.face_detector = MTCNN()

    def _setup_logging(self) -> None:
        """Cấu hình logging."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Phát hiện khuôn mặt trong ảnh và trích xuất vùng mặt."""
        faces = []
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.face_detector.detect_faces(image_rgb)
            for detection in detections:
                x, y, width, height = detection['box']
                x, y = max(0, x), max(0, y)
                face_img = image_rgb[y:y + height, x:x + width]
                faces.append(face_img)
        except Exception as e:
            self.logger.error(f"❌ Lỗi phát hiện khuôn mặt: {e}")
        return faces

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Tiền xử lý ảnh để phù hợp với đầu vào của FaceNet."""
        try:
            image = cv2.resize(image, (160, 160))
            image = image.astype(np.float32) / 255.0
            return np.expand_dims(image, axis=0)
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý ảnh: {e}")
            return None

    def get_embeddings(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """Trích xuất embedding từ ảnh chứa khuôn mặt."""
        face_images = self._detect_faces(image)
        if not face_images:
            self.logger.warning("⚠️ Không tìm thấy khuôn mặt nào trong ảnh.")
            return None

        embeddings = []
        for face_image in face_images:
            preprocessed_image = self._preprocess_image(face_image)
            if preprocessed_image is None:
                continue
            try:
                embedding = self.facenet.embeddings(preprocessed_image)[0]
                embeddings.append(embedding)  # Không chuẩn hóa
            except Exception as e:
                self.logger.error(f"❌ Lỗi trích xuất embedding: {e}")
                continue

        return embeddings if embeddings else None

    def recognize_face(self, face_embedding: np.ndarray, database: List[dict]) -> Optional[dict]:
        """
        So sánh embedding với cơ sở dữ liệu để tìm khuôn mặt phù hợp nhất.
        Args:
            face_embedding: Embedding của khuôn mặt cần nhận diện.
            database: Danh sách các khuôn mặt đã lưu trong MongoDB.
        Returns:
            Thông tin của khuôn mặt nhận diện được (nếu có) hoặc None.
        """
        best_match = None
        min_distance = float("inf")
        for face in database:
            stored_embedding = np.array(face["embedding"], dtype=np.float32)
            distance = euclidean_distance(face_embedding, stored_embedding)
            if distance < self.threshold and distance < min_distance:
                min_distance = distance
                best_match = face
        return best_match
