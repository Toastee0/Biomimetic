#!/usr/bin/env python3
"""
Model Manager for Small Specialized Vision Models
Handles loading/unloading of lightweight models to coexist with Mistral-Small-22B
Total VRAM budget: ~700MB
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages small specialized vision models with VRAM monitoring
    
    Models:
    - InsightFace: Face recognition (~200MB)
    - FER: Emotion detection (~100MB)
    - CLIP: Scene understanding (~350MB)
    - Age/Gender: Lightweight models (~50MB)
    """
    
    def __init__(self, config_path: str = "config/vision_models.json"):
        self.config_path = Path(config_path)
        self.models: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"ModelManager initialized on {self.device}")
        if self.device == "cuda":
            self._log_vram_usage()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        default_config = {
            "models": {
                "face_recognition": {
                    "enabled": True,
                    "model_name": "buffalo_l",
                    "vram_mb": 200
                },
                "emotion": {
                    "enabled": True,
                    "model_name": "fer",
                    "vram_mb": 100
                },
                "clip": {
                    "enabled": True,
                    "model_name": "ViT-B/32",
                    "vram_mb": 350
                },
                "age_gender": {
                    "enabled": True,
                    "model_name": "insightface",
                    "vram_mb": 50
                }
            },
            "total_vram_budget_mb": 700
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _log_vram_usage(self):
        """Log current VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    
    def load_face_recognition(self) -> bool:
        """
        Load InsightFace face recognition model
        Returns: True if successful
        """
        if "face_recognition" in self.models:
            logger.info("Face recognition already loaded")
            return True
        
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            logger.info("Loading face recognition model (InsightFace)...")
            
            # Initialize with GPU if available
            app = FaceAnalysis(
                name=self.config["models"]["face_recognition"]["model_name"],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            )
            app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            
            self.models["face_recognition"] = app
            logger.info("✓ Face recognition loaded successfully")
            self._log_vram_usage()
            return True
            
        except ImportError:
            logger.error("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
            return False
        except Exception as e:
            logger.error(f"Failed to load face recognition: {e}")
            return False
    
    def load_emotion_detector(self) -> bool:
        """
        Load FER emotion detection model
        Returns: True if successful
        """
        if "emotion" in self.models:
            logger.info("Emotion detector already loaded")
            return True
        
        try:
            from fer import FER
            
            logger.info("Loading emotion detection model (FER)...")
            
            # Initialize FER with GPU support
            detector = FER(mtcnn=True)  # MTCNN for face detection
            
            self.models["emotion"] = detector
            logger.info("✓ Emotion detector loaded successfully")
            self._log_vram_usage()
            return True
            
        except ImportError:
            logger.error("FER not installed. Run: pip install fer")
            return False
        except Exception as e:
            logger.error(f"Failed to load emotion detector: {e}")
            return False
    
    def load_clip(self) -> bool:
        """
        Load CLIP for general scene understanding
        Returns: True if successful
        """
        if "clip" in self.models:
            logger.info("CLIP already loaded")
            return True
        
        try:
            import clip
            
            logger.info("Loading CLIP model...")
            
            model_name = self.config["models"]["clip"]["model_name"]
            model, preprocess = clip.load(model_name, device=self.device)
            
            self.models["clip"] = {
                "model": model,
                "preprocess": preprocess
            }
            logger.info(f"✓ CLIP ({model_name}) loaded successfully")
            self._log_vram_usage()
            return True
            
        except ImportError:
            logger.error("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
            return False
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            return False
    
    def unload_model(self, model_name: str):
        """Unload a specific model to free VRAM"""
        if model_name in self.models:
            del self.models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"✓ Unloaded {model_name}")
            self._log_vram_usage()
    
    def unload_all(self):
        """Unload all models and clear VRAM"""
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✓ All models unloaded")
        self._log_vram_usage()
    
    def analyze_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze face in image
        
        Args:
            image: numpy array (RGB)
            
        Returns:
            Dict with face data or None
        """
        if "face_recognition" not in self.models:
            logger.warning("Face recognition not loaded")
            return None
        
        try:
            app = self.models["face_recognition"]
            faces = app.get(image)
            
            if len(faces) == 0:
                return None
            
            # Return first (largest) face
            face = faces[0]
            
            return {
                "embedding": face.embedding.tolist(),  # 512-dim vector
                "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                "det_score": float(face.det_score),  # Detection confidence
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": face.gender if hasattr(face, 'gender') else None,
                "landmark": face.landmark.tolist() if hasattr(face, 'landmark') else None
            }
            
        except Exception as e:
            logger.error(f"Face analysis failed: {e}")
            return None
    
    def detect_emotion(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Detect emotions in image
        
        Args:
            image: numpy array (RGB)
            
        Returns:
            Dict with emotion scores or None
        """
        if "emotion" not in self.models:
            logger.warning("Emotion detector not loaded")
            return None
        
        try:
            detector = self.models["emotion"]
            
            # FER expects BGR format
            image_bgr = image[:, :, ::-1]
            
            emotions = detector.detect_emotions(image_bgr)
            
            if len(emotions) == 0:
                return None
            
            # Return first face emotions
            return emotions[0]["emotions"]
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return None
    
    def analyze_scene_clip(self, image: Image.Image, text_queries: list) -> Optional[Dict[str, float]]:
        """
        Analyze scene using CLIP
        
        Args:
            image: PIL Image
            text_queries: List of text descriptions to compare
            
        Returns:
            Dict mapping queries to similarity scores
        """
        if "clip" not in self.models:
            logger.warning("CLIP not loaded")
            return None
        
        try:
            import clip
            
            model = self.models["clip"]["model"]
            preprocess = self.models["clip"]["preprocess"]
            
            # Preprocess image
            image_input = preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_tokens = clip.tokenize(text_queries).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Return as dict
            return {
                query: float(score)
                for query, score in zip(text_queries, similarity[0].cpu().numpy())
            }
            
        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return None
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self.models.keys())
    
    def estimate_vram_usage(self) -> int:
        """Estimate current VRAM usage in MB"""
        total_mb = 0
        for model_name in self.models.keys():
            if model_name in self.config["models"]:
                total_mb += self.config["models"][model_name].get("vram_mb", 0)
        return total_mb


def main():
    """Test the model manager"""
    manager = ModelManager()
    
    print("\n=== Model Manager Test ===\n")
    
    # Test loading models
    print("1. Loading face recognition...")
    if manager.load_face_recognition():
        print("   ✓ Success")
    else:
        print("   ✗ Failed (install with: pip install insightface onnxruntime-gpu)")
    
    print("\n2. Loading emotion detector...")
    if manager.load_emotion_detector():
        print("   ✓ Success")
    else:
        print("   ✗ Failed (install with: pip install fer)")
    
    print("\n3. Loading CLIP...")
    if manager.load_clip():
        print("   ✓ Success")
    else:
        print("   ✗ Failed (install with: pip install git+https://github.com/openai/CLIP.git)")
    
    print(f"\n4. Loaded models: {manager.get_loaded_models()}")
    print(f"5. Estimated VRAM usage: {manager.estimate_vram_usage()}MB")
    
    print("\n6. Unloading all models...")
    manager.unload_all()
    print(f"   Loaded models: {manager.get_loaded_models()}")
    
    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    main()
