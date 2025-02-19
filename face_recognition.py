import numpy as np
import logging
import os
import tensorflow as tf
import cv2
from typing import Optional
import mediapipe as mp

class FaceRecognitionSystem:
    def __init__(self, model_path: str = "models/arcface_model.tflite", threshold: float = 0.5):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt sử dụng mô hình TF Lite.
        Args:
            model_path: Đường dẫn lưu mô hình TF Lite.
            threshold: Ngưỡng nhận diện.
        """
        self.threshold = threshold
        self.model_path = model_path
        self.model = None  # Bộ nhớ đệm cho mô hình TF Lite
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._setup_logging()
        self._load_model()
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        
    def _detect_faces(self, image: np.ndarray) -> list:
        """
        Phát hiện tất cả các khuôn mặt trong ảnh và trả về danh sách các vùng chứa khuôn mặt.
        Mỗi khuôn mặt được cắt ra dưới dạng một numpy array.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)
        faces = []
        
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                # Đảm bảo tọa độ không vượt ngoài biên ảnh
                x, y = max(0, x), max(0, y)
                face_img = image[y:y + height, x:x + width]
                faces.append(face_img)
        return faces

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
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.logger.info("✅ Mô hình TensorFlow Lite đã được tải thành công.")
            except Exception as e:
                self.logger.error(f"❌ Lỗi tải mô hình: {e}")
                raise

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Tiền xử lý ảnh đầu vào trước khi đưa vào mô hình.
        """
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
            input_size = self.input_details[0]['shape'][1:3]  # Lấy kích thước đầu vào từ mô hình
            image = cv2.resize(image, tuple(input_size))
            image = image.astype(np.float32) / 255.0  # Chuẩn hóa về khoảng [0,1]
            return np.expand_dims(image, axis=0)
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý ảnh: {e}")
            return None

    def get_embeddings(self, image: np.ndarray) -> Optional[list]:
        """
        Trích xuất embedding từ tất cả các khuôn mặt được phát hiện trong ảnh.
        Trả về danh sách các embedding đã được chuẩn hóa.
        Nếu không phát hiện được khuôn mặt nào, trả về None.
        """
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
                self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
                self.interpreter.invoke()
                embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                # Chuẩn hóa embedding và thêm vào danh sách
                normalized_embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(normalized_embedding)
            except Exception as e:
                self.logger.error(f"❌ Lỗi trích xuất embedding: {e}")
                continue

        if not embeddings:
            return None
        return embeddings

    def compare_faces(self, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> bool:
        """So sánh hai embeddings để xác thực khuôn mặt."""
        try:
            if embedding1 is None or embedding2 is None:
                return False
            similarity = np.dot(embedding1, embedding2)
            return similarity > self.threshold
        except Exception as e:
            self.logger.error(f"❌ Lỗi so sánh khuôn mặt: {e}")
            return False