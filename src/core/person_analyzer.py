#!/usr/bin/env python3
"""
Person Analyzer - Specialized analysis for person detections
Matches faces against contacts, estimates demographics, detects emotions
"""

import sys
import json
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.model_manager import ModelManager


class PersonAnalyzer:
    """
    Analyzes people in images using specialized models
    
    Capabilities:
    - Face matching against known contacts
    - Age estimation
    - Gender detection
    - Emotion recognition
    - Updates contact memory with visual observations
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.contacts_path = Path("/home/toastee/BioMimeticAi/data/contacts.json")
        self.contacts = self._load_contacts()
        
        print("[PERSON ANALYZER] Initialized")
    
    def _load_contacts(self) -> Dict[str, Any]:
        """Load known contacts with face embeddings"""
        if not self.contacts_path.exists():
            return {}
        
        try:
            with open(self.contacts_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ANALYZER ERROR] Failed to load contacts: {e}")
            return {}
    
    def _save_contacts(self):
        """Save updated contacts"""
        try:
            self.contacts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.contacts_path, 'w') as f:
                json.dump(self.contacts, f, indent=2)
        except Exception as e:
            print(f"[ANALYZER ERROR] Failed to save contacts: {e}")
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def analyze_person(self, image: np.ndarray, yolo_bbox: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Complete analysis of a person in an image
        
        Args:
            image: numpy array (RGB)
            yolo_bbox: Optional [x1, y1, x2, y2] to crop region
            
        Returns:
            Dict with all analysis results
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "face_detected": False,
            "identity": None,
            "confidence": 0.0,
            "age": None,
            "gender": None,
            "emotion": None,
            "bbox": yolo_bbox
        }
        
        # Crop to YOLO bbox if provided
        if yolo_bbox:
            x1, y1, x2, y2 = map(int, yolo_bbox)
            image_crop = image[y1:y2, x1:x2]
        else:
            image_crop = image
        
        # Face analysis (includes age/gender from InsightFace)
        face_data = self.model_manager.analyze_face(image_crop)
        
        if face_data:
            result["face_detected"] = True
            result["age"] = face_data.get("age")
            result["gender"] = face_data.get("gender")
            
            # Match against known contacts
            if "embedding" in face_data:
                match = self._match_face(face_data["embedding"])
                if match:
                    result["identity"] = match["name"]
                    result["confidence"] = match["confidence"]
                    
                    # Update contact's last seen
                    self._update_contact_observation(match["name"], face_data)
        
        # Emotion detection
        emotions = self.model_manager.detect_emotion(image_crop)
        if emotions:
            # Get dominant emotion
            dominant = max(emotions.items(), key=lambda x: x[1])
            result["emotion"] = {
                "dominant": dominant[0],
                "confidence": dominant[1],
                "all": emotions
            }
        
        return result
    
    def _match_face(self, embedding: List[float], threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Match face embedding against known contacts
        
        Args:
            embedding: Face embedding vector (512-dim)
            threshold: Minimum similarity for match (0.6 = 60%)
            
        Returns:
            Dict with name and confidence, or None
        """
        if not self.contacts:
            return None
        
        embedding_np = np.array(embedding)
        best_match = None
        best_score = threshold
        
        for name, contact_data in self.contacts.items():
            if "face_embeddings" not in contact_data:
                continue
            
            # Compare with all stored embeddings for this contact
            for stored_embedding in contact_data["face_embeddings"]:
                stored_np = np.array(stored_embedding)
                similarity = self._cosine_similarity(embedding_np, stored_np)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        "name": name,
                        "confidence": similarity
                    }
        
        return best_match
    
    def _update_contact_observation(self, name: str, face_data: Dict[str, Any]):
        """Update contact memory with new observation"""
        if name not in self.contacts:
            self.contacts[name] = {
                "face_embeddings": [],
                "observations": []
            }
        
        contact = self.contacts[name]
        
        # Add new embedding (keep max 5 for averaging)
        if "face_embeddings" not in contact:
            contact["face_embeddings"] = []
        
        if len(contact["face_embeddings"]) < 5:
            contact["face_embeddings"].append(face_data["embedding"])
        
        # Record observation
        if "observations" not in contact:
            contact["observations"] = []
        
        obs = {
            "timestamp": datetime.now().isoformat(),
            "age": face_data.get("age"),
            "gender": face_data.get("gender"),
            "bbox": face_data.get("bbox")
        }
        
        contact["observations"].append(obs)
        
        # Keep only recent 50 observations
        if len(contact["observations"]) > 50:
            contact["observations"] = contact["observations"][-50:]
        
        self._save_contacts()
    
    def register_new_contact(self, name: str, image: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Register a new contact with face embedding
        
        Args:
            name: Person's name
            image: Image containing their face
            metadata: Optional additional info
            
        Returns:
            True if successful
        """
        face_data = self.model_manager.analyze_face(image)
        
        if not face_data or "embedding" not in face_data:
            print(f"[ANALYZER] No face found in image for {name}")
            return False
        
        self.contacts[name] = {
            "name": name,
            "face_embeddings": [face_data["embedding"]],
            "age": face_data.get("age"),
            "gender": face_data.get("gender"),
            "registered": datetime.now().isoformat(),
            "observations": [],
            "metadata": metadata or {}
        }
        
        self._save_contacts()
        print(f"[ANALYZER] ✓ Registered contact: {name}")
        return True
    
    def get_contact_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get stored information about a contact"""
        return self.contacts.get(name)
    
    def list_contacts(self) -> List[str]:
        """Get list of all registered contacts"""
        return list(self.contacts.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about person observations"""
        total_contacts = len(self.contacts)
        total_observations = sum(
            len(contact.get("observations", []))
            for contact in self.contacts.values()
        )
        
        return {
            "total_contacts": total_contacts,
            "total_observations": total_observations,
            "contacts": list(self.contacts.keys())
        }


def main():
    """Test person analyzer"""
    print("\n=== Person Analyzer Test ===\n")
    
    # Initialize
    analyzer = PersonAnalyzer()
    
    # Check statistics
    stats = analyzer.get_statistics()
    print(f"Contacts registered: {stats['total_contacts']}")
    print(f"Total observations: {stats['total_observations']}")
    
    if stats['contacts']:
        print(f"Known contacts: {', '.join(stats['contacts'])}")
    else:
        print("No contacts registered yet")
    
    print("\n=== To register a new contact ===")
    print("from PIL import Image")
    print("import numpy as np")
    print("image = np.array(Image.open('path/to/photo.jpg'))")
    print("analyzer.register_new_contact('Name', image)")
    
    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    main()
