import numpy as np
import logging
import os
import tensorflow as tf
from typing import Optional
from deepface import DeepFace
import cv2

class FaceRecognitionSystem:
    def __init__(self, model_name: str = "Facenet", threshold: float = 0.5):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt
        Args:
            model_name: Tên mô hình (mặc định: Facenet)
            threshold: Ngưỡng nhận diện (mặc định: 0.5)
        """
        self.threshold = threshold
        self.model_name = model_name
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Cấu hình logging."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

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
        """Trích xuất embedding từ ảnh khuôn mặt."""
        preprocessed_image = self._preprocess_image(image)
        if preprocessed_image is None:
            return None
        
        temp_path = 'temp_face.jpg'
        try:
            # Save the preprocessed image temporarily
            cv2.imwrite(temp_path, preprocessed_image)
            
            try:
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                return np.array(embedding[0]['embedding']) if embedding else None
            
            except ValueError as ve:
                if "Invalid model_name" in str(ve):
                    self.logger.error(f"❌ Model name '{self.model_name}' không hợp lệ. Hãy sử dụng tên mô hình khả dụng như 'Facenet', 'VGG-Face', v.v.")
                    # Fallback to default model if possible
                    if self.model_name != "Facenet":
                        self.logger.info("🔄 Thử lại với mô hình mặc định 'Facenet'")
                        self.model_name = "Facenet"
                        return self.get_embedding(image)
                else:
                    self.logger.error(f"❌ Lỗi giá trị: {ve}")
                return None
            
            except Exception as e:
                self.logger.error(f"❌ Lỗi trích xuất embedding: {e}")
                return None
            
        finally:
            # Clean up regardless of success or failure
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