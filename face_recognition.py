import numpy as np
import logging
import os
import tensorflow as tf
import cv2
from typing import Optional

class FaceRecognitionSystem:
    def __init__(self, model_path: str = "models/arcface_weights.tflite", threshold: float = 0.5):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt sử dụng mô hình TF Lite.
        Args:
            model_path: Đường dẫn lưu mô hình TF Lite.
            threshold: Ngưỡng nhận diện.
        """
        self.threshold = threshold
        self.model_path = model_path
        self.model = None  # Bộ nhớ đệm cho mô hình TF Lite
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Cấu hình logging."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _load_model(self) -> None:
        """Tải mô hình TF Lite vào bộ nhớ nếu chưa được tải."""
        if self.model is None:
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ Không tìm thấy mô hình tại {self.model_path}.")
                raise FileNotFoundError("Mô hình không tồn tại!")
            try:
                self.logger.info("🔄 Đang tải mô hình TensorFlow Lite vào bộ nhớ...")
                interpreter = tf.lite.Interpreter(model_path=self.model_path)
                interpreter.allocate_tensors()
                self.model = interpreter
                self.logger.info("✅ Mô hình TensorFlow Lite đã được tải thành công.")
            except Exception as e:
                self.logger.error(f"❌ Lỗi tải mô hình: {e}")
                raise

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Tiền xử lý ảnh đầu vào:
           - Nếu ảnh có 3 kênh (BGR) chuyển sang RGB.
           - Resize ảnh về kích thước (112, 112).
        """
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(image, (112, 112))
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý ảnh: {e}")
            return None

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Trích xuất embedding từ ảnh khuôn mặt sử dụng mô hình TF Lite.
        Args:
            image: Ảnh khuôn mặt đầu vào dưới dạng numpy.ndarray.
        Returns:
            Embedding vector nếu thành công, ngược lại trả về None.
        """
        # Tải mô hình nếu chưa được tải
        self._load_model()

        preprocessed_image = self._preprocess_image(image)
        if preprocessed_image is None:
            return None

        try:
            # Lấy thông tin input và output từ mô hình TF Lite
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            # Chuẩn bị dữ liệu đầu vào: mở rộng chiều và chuyển về kiểu float32
            input_data = np.expand_dims(preprocessed_image, axis=0).astype(np.float32)

            # Đưa dữ liệu vào mô hình và chạy inference
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()

            # Lấy kết quả embedding
            embedding = self.model.get_tensor(output_details[0]['index'])[0]
            return embedding
        except Exception as e:
            self.logger.error(f"❌ Lỗi trích xuất embedding: {e}")
            return None

    def compare_faces(self, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> bool:
        """
        So sánh hai embeddings để xác thực khuôn mặt.
        Args:
            embedding1: Embedding của khuôn mặt thứ nhất.
            embedding2: Embedding của khuôn mặt thứ hai.
        Returns:
            True nếu similarity vượt ngưỡng, ngược lại False.
        """
        try:
            if embedding1 is None or embedding2 is None:
                return False
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return similarity > self.threshold
        except Exception as e:
            self.logger.error(f"❌ Lỗi so sánh khuôn mặt: {e}")
            return False
