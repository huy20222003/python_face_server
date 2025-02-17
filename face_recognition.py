import numpy as np
import logging
import os
import tensorflow as tf
from typing import Optional
from deepface import DeepFace
import cv2
import gdown  # Thêm thư viện để tải file từ Google Drive

class FaceRecognitionSystem:
    def __init__(self, model_name: str = "FaceNet", threshold: float = 0.5, model_path: str = "models/facenet_keras.h5"):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt
        Args:
            model_name: Tên mô hình (mặc định: FaceNet)
            threshold: Ngưỡng nhận diện (mặc định: 0.5)
            model_path: Đường dẫn đến mô hình đã lưu
        """
        self.threshold = threshold
        self.model_name = model_name
        self.model_path = model_path
        self.model_url = "https://drive.google.com/file/d/1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1/view?usp=sharing"  # Cập nhật Google Drive ID
        self.model = None  # Lazy Loading
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Cấu hình logging."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _download_model(self) -> None:
        """Tải mô hình từ Google Drive nếu chưa có."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.logger.info("📥 Đang tải mô hình từ Google Drive (đồng bộ)...")
            try:
                # Sử dụng fuzzy=True để xử lý link chia sẻ của Google Drive
                gdown.download(url=self.model_url, output=self.model_path, quiet=False, fuzzy=True)
                self.logger.info("✅ Model đã tải thành công!")
                
                # (Tùy chọn) Kiểm tra kích thước file hoặc trạng thái file sau khi tải về
                if os.path.getsize(self.model_path) < 100 * 1024 * 1024:
                    self.logger.warning("⚠️ Kích thước file tải về có vẻ không đúng, vui lòng kiểm tra lại.")
            except Exception as e:
                self.logger.error(f"❌ Lỗi tải model: {e}")
                raise

    def _load_model(self) -> None:
        """Load mô hình từ file, nếu chưa có thì tải trước rồi load."""
        if self.model is None:
            try:
                if not os.path.exists(self.model_path):
                    self.logger.info(f"⚠️ Không tìm thấy {self.model_path}, đang tải model...")
                    self._download_model()
                
                self.logger.info(f"🔄 Đang tải mô hình từ {self.model_path}...")
                # Chờ một chút nếu cần thiết để đảm bảo file đã hoàn toàn ghi xong (trường hợp hệ thống file chậm)
                while not os.path.exists(self.model_path):
                    pass

                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info(f"✅ Mô hình {self.model_name} đã tải xong")
            except Exception as e:
                self.logger.error(f"❌ Lỗi tải mô hình: {e}")
                raise
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Tiền xử lý ảnh đầu vào."""
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(image, (160, 160)).astype(np.float32)  # Resize nhỏ hơn để tiết kiệm RAM
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý ảnh: {e}")
            return None
    
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Trích xuất embedding từ ảnh khuôn mặt."""
        self._load_model()
        preprocessed_image = self._preprocess_image(image)
        if preprocessed_image is None:
            return None
        try:
            embedding = DeepFace.represent(
                img_path=preprocessed_image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            return np.array(embedding[0]['embedding']) if embedding else None
        except Exception as e:
            self.logger.error(f"❌ Lỗi trích xuất embedding: {e}")
            return None
    
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
