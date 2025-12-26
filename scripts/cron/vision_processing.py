#!/usr/bin/env python3
"""
Vision Processing Cortex - Processes queued snapshots every 30 minutes
Runs as a cron job to analyze queued snapshots with specialized models

NOW WITH: Database integration for autonomous vision pool
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import sqlite3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.model_manager import ModelManager
from src.core.person_analyzer import PersonAnalyzer
from src.memory.episodic import EpisodicMemory


class VisionProcessingCortex:
    """
    Batch processes queued vision snapshots
    
    Processing steps:
    1. Load queued snapshots
    2. Load specialized models (face, emotion, CLIP)
    3. Process each snapshot based on detected object
    4. Update episodic memory with rich analysis
    5. Mark as processed
    6. Unload models to free VRAM
    """
    
    def __init__(self):
        self.queue_path = Path("/home/toastee/BioMimeticAi/data/vision/snapshot_queue.json")
        self.processed_path = Path("/home/toastee/BioMimeticAi/data/vision/processed.json")
        self.db_path = Path("/home/toastee/BioMimeticAi/data/biomim.db")

        self.model_manager = ModelManager()
        self.person_analyzer = PersonAnalyzer(self.model_manager)
        self.episodic = EpisodicMemory()

        print("[VISION CORTEX] Initialized")
    
    def get_db_connection(self):
        """Get database connection with WAL mode and timeout"""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def load_queue(self) -> List[Dict[str, Any]]:
        """
        Load unprocessed snapshots from queue

        NOW ALSO: Query database for unprocessed segments
        """
        queue = []

        # Load from JSON queue (legacy support)
        if self.queue_path.exists():
            try:
                with open(self.queue_path, 'r') as f:
                    json_queue = json.load(f)

                # Filter unprocessed
                unprocessed = [item for item in json_queue if not item.get("processed", False)]
                queue.extend(unprocessed)
                print(f"[VISION CORTEX] Found {len(unprocessed)} unprocessed snapshots in JSON queue")

            except Exception as e:
                print(f"[VISION CORTEX ERROR] Failed to load JSON queue: {e}")

        # Load from database (new autonomous pool)
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT segment_id, file_path, event_type, camera_id,
                           urgency_score, yolo_metadata, timestamp
                    FROM vision_segments
                    WHERE processed = 0
                    ORDER BY urgency_score DESC, timestamp ASC
                """)

                db_segments = cursor.fetchall()

                for row in db_segments:
                    # Convert database row to queue item format
                    yolo_metadata = json.loads(row['yolo_metadata']) if row['yolo_metadata'] else None

                    queue.append({
                        "segment_id": row['segment_id'],
                        "snapshot_path": row['file_path'],
                        "event_type": row['event_type'],
                        "detected_object": yolo_metadata.get('class', 'unknown') if yolo_metadata else 'unknown',
                        "timestamp": row['timestamp'] * 1000,  # Convert s to ms
                        "yolo_detection": yolo_metadata,
                        "urgency_score": row['urgency_score'],
                        "from_database": True  # Flag to distinguish from JSON queue
                    })

                print(f"[VISION CORTEX] Found {len(db_segments)} unprocessed segments in database")

        except Exception as e:
            print(f"[VISION CORTEX ERROR] Failed to load database segments: {e}")

        print(f"[VISION CORTEX] Total unprocessed: {len(queue)}")
        return queue
    
    def save_queue(self, queue: List[Dict[str, Any]]):
        """Save updated queue"""
        try:
            with open(self.queue_path, 'w') as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            print(f"[VISION CORTEX ERROR] Failed to save queue: {e}")
    
    def save_processed(self, item: Dict[str, Any], analysis: Dict[str, Any]):
        """Save processed item to history"""
        try:
            # Load existing processed items
            if self.processed_path.exists():
                with open(self.processed_path, 'r') as f:
                    processed = json.load(f)
            else:
                processed = []
            
            # Add new item
            processed.append({
                "queue_item": item,
                "analysis": analysis,
                "processed_at": datetime.now().isoformat()
            })
            
            # Keep only recent 1000 items
            if len(processed) > 1000:
                processed = processed[-1000:]
            
            # Save
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.processed_path, 'w') as f:
                json.dump(processed, f, indent=2)
                
        except Exception as e:
            print(f"[VISION CORTEX ERROR] Failed to save processed item: {e}")
    
    def process_person_detection(self, snapshot_path: str, yolo_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process person detection with specialized models
        
        Args:
            snapshot_path: Path to snapshot image
            yolo_detection: YOLO metadata (bbox, confidence, etc.)
            
        Returns:
            Complete analysis results
        """
        print(f"[VISION CORTEX] Processing person detection...")
        
        try:
            # Load image
            image = Image.open(snapshot_path)
            image_np = np.array(image)
            
            # Extract bbox if available
            bbox = yolo_detection.get('bbox') if yolo_detection else None
            
            # Analyze person
            analysis = self.person_analyzer.analyze_person(image_np, bbox)
            
            # Add YOLO metadata
            analysis["yolo"] = yolo_detection
            
            return analysis
            
        except Exception as e:
            print(f"[VISION CORTEX ERROR] Person analysis failed: {e}")
            return {"error": str(e)}
    
    def process_general_scene(self, snapshot_path: str) -> Dict[str, Any]:
        """
        Process general scene with CLIP
        
        Args:
            snapshot_path: Path to snapshot image
            
        Returns:
            Scene analysis results
        """
        print(f"[VISION CORTEX] Processing general scene...")
        
        try:
            # Load CLIP
            if not self.model_manager.load_clip():
                return {"error": "CLIP not available"}
            
            # Load image
            image = Image.open(snapshot_path)
            
            # Define scene queries
            queries = [
                "a person entering a room",
                "a person leaving a room",
                "an empty room",
                "multiple people in a room",
                "a car in a driveway",
                "a pet animal",
                "outdoor scene",
                "indoor scene"
            ]
            
            # Analyze with CLIP
            scores = self.model_manager.analyze_scene_clip(image, queries)
            
            if scores:
                # Get top match
                top_match = max(scores.items(), key=lambda x: x[1])
                
                return {
                    "scene_type": top_match[0],
                    "confidence": top_match[1],
                    "all_scores": scores
                }
            else:
                return {"error": "CLIP analysis failed"}
                
        except Exception as e:
            print(f"[VISION CORTEX ERROR] Scene analysis failed: {e}")
            return {"error": str(e)}
    
    def process_queue(self):
        """Main processing loop"""
        print("\n" + "="*60)
        print(f"[VISION CORTEX] Starting processing run at {datetime.now().isoformat()}")
        print("="*60 + "\n")
        
        # Load queue
        queue = self.load_queue()
        
        if not queue:
            print("[VISION CORTEX] No items to process. Exiting.")
            return
        
        # Load all models upfront
        print("[VISION CORTEX] Loading models...")
        models_loaded = {
            "face_recognition": self.model_manager.load_face_recognition(),
            "emotion": self.model_manager.load_emotion_detector(),
            "clip": self.model_manager.load_clip()
        }
        
        print(f"[VISION CORTEX] Models loaded: {[k for k, v in models_loaded.items() if v]}")
        
        # Process each item
        processed_count = 0
        for i, item in enumerate(queue, 1):
            print(f"\n[VISION CORTEX] Processing {i}/{len(queue)}: {item['detected_object']}")
            
            snapshot_path = item["snapshot_path"]
            
            # Check if snapshot exists
            if not Path(snapshot_path).exists():
                print(f"[VISION CORTEX] ✗ Snapshot not found: {snapshot_path}")
                item["processed"] = True
                item["error"] = "Snapshot file not found"
                continue
            
            # Process based on detected object
            detected_object = item["detected_object"].lower()
            yolo_detection = item.get("yolo_detection")
            
            if detected_object == "person":
                analysis = self.process_person_detection(snapshot_path, yolo_detection)
            else:
                # General scene analysis for non-person objects
                analysis = self.process_general_scene(snapshot_path)
            
            # Calculate salience based on analysis
            salience = self._calculate_salience(analysis, item)
            
            # Store in episodic memory
            self._store_episode(item, analysis, salience)
            
            # Mark as processed
            item["processed"] = True
            item["processed_at"] = datetime.now().isoformat()
            item["analysis_summary"] = self._summarize_analysis(analysis)

            # Save to processed history
            self.save_processed(item, analysis)

            # Update database if this segment came from database
            if item.get("from_database"):
                self._update_database_segment(item["segment_id"], analysis)

            processed_count += 1
            print(f"[VISION CORTEX] ✓ Processed ({processed_count}/{len(queue)})")
        
        # Save updated queue
        self.save_queue(queue)
        
        # Unload all models to free VRAM
        print("\n[VISION CORTEX] Unloading models...")
        self.model_manager.unload_all()
        
        print("\n" + "="*60)
        print(f"[VISION CORTEX] Processing complete: {processed_count} items")
        print("="*60 + "\n")
    
    def _calculate_salience(self, analysis: Dict[str, Any], item: Dict[str, Any]) -> float:
        """Calculate salience score for episodic memory"""
        base_salience = 0.7
        
        # Higher salience for recognized people
        if analysis.get("identity"):
            base_salience += 0.2
        
        # Higher salience for strong emotions
        emotion = analysis.get("emotion")
        if emotion and emotion.get("dominant") in ["angry", "surprised", "fearful"]:
            base_salience += 0.1
        
        # Higher salience for entrance events
        if item.get("event_type") == "entrance":
            base_salience += 0.1
        
        return min(base_salience, 1.0)
    
    def _summarize_analysis(self, analysis: Dict[str, Any]) -> str:
        """Create human-readable summary of analysis"""
        parts = []
        
        if analysis.get("identity"):
            parts.append(f"Identified: {analysis['identity']} (conf: {analysis.get('confidence', 0):.2f})")
        elif analysis.get("face_detected"):
            parts.append("Unknown person")
        
        if analysis.get("age"):
            parts.append(f"Age: ~{analysis['age']}")
        
        if analysis.get("gender"):
            parts.append(f"Gender: {analysis['gender']}")
        
        if analysis.get("emotion"):
            emotion = analysis["emotion"]
            parts.append(f"Emotion: {emotion.get('dominant')} ({emotion.get('confidence', 0):.2f})")
        
        if analysis.get("scene_type"):
            parts.append(f"Scene: {analysis['scene_type']}")
        
        if analysis.get("error"):
            parts.append(f"Error: {analysis['error']}")
        
        return " | ".join(parts) if parts else "No analysis available"
    
    def _store_episode(self, item: Dict[str, Any], analysis: Dict[str, Any], salience: float):
        """Store analysis in episodic memory"""
        try:
            event_type = item.get("event_type", "unknown")
            detected_object = item.get("detected_object", "unknown")
            summary = self._summarize_analysis(analysis)

            message = f"Vision analysis: {detected_object} {event_type} - {summary}"

            self.episodic.store_episode(
                user_id="system_vision",
                username="VisionCortex",
                user_message=message,
                bot_response=json.dumps(analysis, default=str),
                hemisphere="cognitive",
                salience_score=salience
            )

            print(f"[VISION CORTEX] Stored in episodic memory (salience: {salience:.2f})")

        except Exception as e:
            print(f"[VISION CORTEX ERROR] Failed to store episode: {e}")

    def _update_database_segment(self, segment_id: int, analysis: Dict[str, Any]):
        """
        Update database segment after processing

        Args:
            segment_id: Database segment ID
            analysis: Complete analysis results
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()

                # Mark segment as processed
                cursor.execute("""
                    UPDATE vision_segments
                    SET processed = 1, processed_at = ?
                    WHERE segment_id = ?
                """, (int(time.time()), segment_id))

                # Add analysis tags to segment_tags table
                timestamp = int(time.time())

                # Emotion tag (if person detected)
                if analysis.get("emotion"):
                    emotion = analysis["emotion"]
                    cursor.execute("""
                        INSERT INTO segment_tags (
                            segment_id, tag_type, tag_value, confidence,
                            model_name, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        segment_id,
                        'emotion',
                        emotion.get('dominant', 'unknown'),
                        emotion.get('confidence', 0.0),
                        'fer',
                        timestamp
                    ))

                # Age/gender tags (if available)
                if analysis.get("age"):
                    cursor.execute("""
                        INSERT INTO segment_tags (
                            segment_id, tag_type, tag_value, confidence,
                            metadata_json, model_name, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        segment_id,
                        'custom',
                        f"age_{analysis['age']}",
                        1.0,
                        json.dumps({"age": analysis['age'], "gender": analysis.get('gender')}),
                        'insightface',
                        timestamp
                    ))

                # Scene tag (from CLIP)
                if analysis.get("scene_type"):
                    cursor.execute("""
                        INSERT INTO segment_tags (
                            segment_id, tag_type, tag_value, confidence,
                            model_name, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        segment_id,
                        'scene',
                        analysis['scene_type'],
                        analysis.get('confidence', 0.0),
                        'clip_vit_b32',
                        timestamp
                    ))

                # Identity tag (if recognized)
                if analysis.get("identity"):
                    cursor.execute("""
                        INSERT INTO segment_tags (
                            segment_id, tag_type, tag_value, confidence,
                            model_name, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        segment_id,
                        'custom',
                        f"identity_{analysis['identity']}",
                        analysis.get('confidence', 0.0),
                        'insightface',
                        timestamp
                    ))

                conn.commit()
                print(f"[DATABASE] Updated segment {segment_id} with analysis tags")

        except Exception as e:
            print(f"[DATABASE ERROR] Failed to update segment: {e}")


if __name__ == "__main__":
    cortex = VisionProcessingCortex()
    cortex.process_queue()

