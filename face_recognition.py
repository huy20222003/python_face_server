import numpy as np
import logging
import os
import tensorflow as tf
from typing import Optional
import cv2
from deepface.DeepFace import build_model

class FaceRecognitionSystem:
    def __init__(self, model_path: str = "models/facenet_model.h5", threshold: float = 0.5):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt.
        Args:
            model_path: Đường dẫn lưu mô hình cục bộ.
            threshold: Ngưỡng nhận diện.
        """
        self.threshold = threshold
        self.model_path = model_path
        self._setup_logging()
        self._load_model()

    def _setup_logging(self) -> None:
        """Cấu hình logging."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _load_model(self) -> None:
        """Tải mô hình Facenet một lần duy nhất."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"❌ Không tìm thấy mô hình tại {self.model_path}. Vui lòng tải mô hình trước khi deploy.")
            raise FileNotFoundError("Mô hình không tồn tại!")
        try:
            self.logger.info("🔄 Đang tải mô hình vào bộ nhớ...")
            self.model = build_model("Facenet")
            self.model.load_weights(self.model_path)
            self.logger.info("✅ Mô hình đã được tải thành công.")
        except Exception as e:
            self.logger.error(f"❌ Lỗi tải mô hình: {e}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Tiền xử lý ảnh đầu vào."""
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(image, (160, 160))
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý ảnh: {e}")
            return None
    
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Trích xuất embedding từ ảnh khuôn mặt sử dụng mô hình đã tải."""
        preprocessed_image = self._preprocess_image(image)
        if preprocessed_image is None:
            return None

        temp_path = 'temp_face.jpg'
        try:
            cv2.imwrite(temp_path, preprocessed_image)
            embedding = build_model("Facenet").predict(np.expand_dims(preprocessed_image, axis=0))[0]
            return embedding
        except Exception as e:
            self.logger.error(f"❌ Lỗi trích xuất embedding: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def compare_faces(self, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> bool:
        """So sánh hai embeddings để xác thực khuôn mặt."""
        try:
            if embedding1 is None or embedding2 is None:
                return False
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return similarity > self.threshold
        except Exception as e:
            self.logger.error(f"❌ Lỗi so sánh khuôn mặt: {e}")
            return False
