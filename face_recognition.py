import numpy as np
import logging
from typing import Optional, List, Union
import tensorflow as tf
from deepface import DeepFace
from pathlib import Path

class FaceRecognitionSystem:
    def __init__(self, model_path: Union[str, Path], threshold: float = 0.5):
        """
        Initialize the face recognition system.
        
        Args:
            model_path: Path to the FaceNet model weights
            threshold: Similarity threshold for face comparison
        """
        self.threshold = threshold
        self.model = None
        self._setup_logging()
        self._load_model(model_path)
    
    def _setup_logging(self) -> None:
        """Configure logging with appropriate format and level."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: Union[str, Path]) -> None:
        """Load the FaceNet model from the specified path."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = tf.keras.models.load_model(str(model_path))
            self.logger.info("✅ FaceNet model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def get_embedding(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Generate face embedding from input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face embedding vector if successful, None otherwise
        """
        try:
            if self.model is None:
                self.logger.error("❌ Model not available")
                return None
            
            result = DeepFace.represent(
                image,
                model=self.model,
                enforce_detection=False
            )
            
            if result and isinstance(result, list) and "embedding" in result[0]:
                return result[0]["embedding"]
            
            self.logger.error("❌ No embedding found in DeepFace result")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating embedding: {str(e)}")
            return None
    
    def compare_faces(self, 
                     embedding1: Optional[List[float]], 
                     embedding2: Optional[List[float]]) -> bool:
        """
        Compare two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            True if faces match according to threshold, False otherwise
        """
        try:
            if embedding1 is None or embedding2 is None:
                self.logger.warning("⚠️ One or both embeddings are None")
                return False
                
            distance = np.linalg.norm(
                np.array(embedding1) - np.array(embedding2)
            )
            self.logger.debug(f"Distance between embeddings: {distance:.4f}")
            
            return distance < self.threshold
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing faces: {str(e)}")
            return False

# Usage example
if __name__ == "__main__":
    MODEL_PATH = Path("models") / "facenet_weights.h5"
    
    try:
        # Initialize the system
        face_system = FaceRecognitionSystem(MODEL_PATH)
        
        # Example usage (assuming you have image1 and image2)
        # embedding1 = face_system.get_embedding(image1)
        # embedding2 = face_system.get_embedding(image2)
        # match = face_system.compare_faces(embedding1, embedding2)
        
    except Exception as e:
        logging.error(f"System initialization failed: {str(e)}")