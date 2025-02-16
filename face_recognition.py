import numpy as np
import logging
from typing import Optional, List
from deepface import DeepFace
import cv2

class FaceRecognitionSystem:
    def __init__(self, model_name: str = "VGG-Face", threshold: float = 0.5):
        """
        Initialize face recognition system using DeepFace's built-in models.
        
        Args:
            model_name: Name of the model to use (default: VGG-Face)
            threshold: Similarity threshold for face comparison (default: 0.5)
        """
        self.threshold = threshold
        self.model_name = model_name
        self._setup_logging()
        self._initialize_model()
    
    def _setup_logging(self) -> None:
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_model(self) -> None:
        """Initialize the face recognition model."""
        try:
            self.logger.info(f"🔄 Initializing {self.model_name} model...")
            # Verify model availability by attempting to build it
            DeepFace.build_model(self.model_name)
            self.logger.info(f"✅ {self.model_name} model initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing model: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess the input image for face detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image or None if preprocessing fails
        """
        try:
            # Convert BGR to RGB if necessary
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to a standard size
            target_size = (224, 224)
            return cv2.resize(image, target_size)
            
        except Exception as e:
            self.logger.error(f"❌ Error preprocessing image: {str(e)}")
            return None
    
    def get_embedding(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Generate face embedding from input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of embedding values or None if face detection fails
        """
        try:
            preprocessed_image = self._preprocess_image(image)
            if preprocessed_image is None:
                return None
            
            # Generate embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=preprocessed_image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(embedding, list) and len(embedding) > 0:
                # Extract the embedding vector from the result
                if isinstance(embedding[0], dict) and 'embedding' in embedding[0]:
                    return embedding[0]['embedding']
                return embedding[0]
                
            self.logger.error("❌ Failed to generate embedding")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating embedding: {str(e)}")
            return None
    
    def compare_faces(self, 
                     embedding1: Optional[List[float]], 
                     embedding2: Optional[List[float]]) -> bool:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Boolean indicating whether faces match
        """
        try:
            if embedding1 is None or embedding2 is None:
                self.logger.warning("⚠️ One or both embeddings are None")
                return False
            
            # Convert lists to numpy arrays if necessary
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
                
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            
            self.logger.debug(f"Face similarity score: {similarity:.4f}")
            return similarity > self.threshold
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing faces: {str(e)}")
            return False