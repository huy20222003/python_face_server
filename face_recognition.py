import numpy as np
import logging
import os
import tensorflow as tf
import cv2
from typing import Optional, List
from mtcnn.mtcnn import MTCNN

class FaceRecognitionSystem:
    def __init__(self, model_path: str = "models/arcface_model.tflite", threshold: float = 0.8):
        """
        Initialize the face recognition system using a TensorFlow Lite model.
        
        Args:
            model_path: Path to the TF Lite model file.
            threshold: Similarity threshold for face verification.
        """
        self.threshold = threshold
        self.model_path = model_path
        self.model = None  # Cache for the TF Lite model (if needed in future)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._setup_logging()
        self._load_model()
        self.face_detector = MTCNN()

    def _setup_logging(self) -> None:
        """Set up the logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate the input image before further processing.
        
        Args:
            image: Input image as a numpy array.
        
        Returns:
            True if the image is valid; False otherwise.
        """
        if image is None:
            self.logger.error("❌ Input image is None.")
            return False
        if not isinstance(image, np.ndarray):
            self.logger.error("❌ Input is not a numpy array.")
            return False
        if image.size == 0:
            self.logger.error("❌ Input image is empty.")
            return False
        if len(image.shape) != 3 or image.shape[2] != 3:
            self.logger.error("❌ Input image must have 3 color channels (BGR/RGB).")
            return False
        return True

    def _load_model(self) -> None:
        """Load the TensorFlow Lite model into memory."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"❌ Model not found at {self.model_path}.")
            raise FileNotFoundError("Model does not exist!")
        try:
            self.logger.info("🔄 Loading TensorFlow Lite model into memory...")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.logger.info("✅ TensorFlow Lite model loaded successfully.")
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess the face image for the model.
        
        Args:
            image: Face image as a numpy array.
            
        Returns:
            Preprocessed image with added batch dimension, or None if an error occurs.
        """
        try:
            # Convert from BGR (OpenCV default) to RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Get the expected input size from the loaded model.
            target_size = tuple(self.input_details[0]['shape'][1:3])
            image_resized = cv2.resize(image_rgb, target_size)
            image_normalized = image_resized.astype(np.float32) / 255.0
            return np.expand_dims(image_normalized, axis=0)
        except Exception as e:
            self.logger.error(f"❌ Error preprocessing image: {e}")
            return None

    def _detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in the provided image.
        
        Args:
            image: Input image as a numpy array.
        
        Returns:
            A list of face region images extracted from the input image.
        """
        faces = []
        try:
            if not self.validate_image(image):
                return faces

            # Create a defensive copy.
            image_copy = image.copy()
            # Convert image from BGR to RGB.
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            detections = self.face_detector.detect_faces(image_rgb)

            if not detections:
                self.logger.info("ℹ️ No faces detected in the image.")
                return faces

            for detection in detections:
                x, y, width, height = detection['box']
                # Add padding to the face region.
                padding = int(min(width, height) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(image_copy.shape[1] - x, width + 2 * padding)
                height = min(image_copy.shape[0] - y, height + 2 * padding)

                face_img = image_copy[y:y + height, x:x + width]
                if face_img.size > 0:
                    faces.append(face_img)
                else:
                    self.logger.warning("⚠️ Empty face region detected.")
        except Exception as e:
            self.logger.error(f"❌ Face detection error: {e}")
            self.logger.exception("Detailed error traceback:")
        return faces

    def get_embeddings(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Extract normalized embeddings from detected face regions.
        
        Args:
            image: Input image as a numpy array.
        
        Returns:
            A list of normalized embeddings, or None if no valid faces are detected.
        """
        face_images = self._detect_faces(image)
        if not face_images:
            self.logger.warning("⚠️ No faces found in the image.")
            return None

        embeddings = []
        for face_image in face_images:
            preprocessed = self._preprocess_image(face_image)
            if preprocessed is None:
                continue
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
                self.interpreter.invoke()
                # Retrieve the embedding vector from model output.
                embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    self.logger.warning("⚠️ Zero norm encountered in embedding, skipping this face.")
                    continue
                normalized_embedding = embedding / norm
                embeddings.append(normalized_embedding)
            except Exception as e:
                self.logger.error(f"❌ Error extracting embedding: {e}")
                continue

        return embeddings if embeddings else None

    def compare_faces(self, embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> bool:
        """
        Compare two embeddings and determine if they belong to the same face.
        
        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.
        
        Returns:
            True if the cosine similarity exceeds the threshold; False otherwise.
        """
        try:
            if embedding1 is None or embedding2 is None:
                self.logger.error("❌ One or both embeddings are None.")
                return False
            similarity = np.dot(embedding1, embedding2)
            return similarity > self.threshold
        except Exception as e:
            self.logger.error(f"❌ Error comparing faces: {e}")
            return False
