#!/usr/bin/env python3
"""
Vision Processing Cortex - Processes queued snapshots every 30 minutes
Runs as a cron job to analyze queued snapshots with specialized models
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from PIL import Image

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
        
        self.model_manager = ModelManager()
        self.person_analyzer = PersonAnalyzer(self.model_manager)
        self.episodic = EpisodicMemory()
        
        print("[VISION CORTEX] Initialized")
    
    def load_queue(self) -> List[Dict[str, Any]]:
        """Load unprocessed snapshots from queue"""
        if not self.queue_path.exists():
            print("[VISION CORTEX] No queue file found")
            return []
        
        try:
            with open(self.queue_path, 'r') as f:
                queue = json.load(f)
            
            # Filter unprocessed
            unprocessed = [item for item in queue if not item.get("processed", False)]
            
            print(f"[VISION CORTEX] Found {len(unprocessed)} unprocessed snapshots")
            return unprocessed
            
        except Exception as e:
            print(f"[VISION CORTEX ERROR] Failed to load queue: {e}")
            return []
    
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


def main():
    """Run vision processing cortex"""
    try:
        cortex = VisionProcessingCortex()
        cortex.process_queue()
    except Exception as e:
        print(f"[VISION CORTEX FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    """Main vision processing loop"""
    start_time = time.time()
    log("[START] Vision Processing Cortex")

    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        episodic = EpisodicMemory()
        items_processed = 0
        errors = []

        # Load snapshot queue
        if not SNAPSHOT_QUEUE_PATH.exists():
            log("[INFO] No snapshot queue found - nothing to process")
            save_state(status="success", items_processed=0)
            return

        with open(SNAPSHOT_QUEUE_PATH, 'r') as f:
            queue = json.load(f)

        # Filter unprocessed snapshots
        unprocessed = [item for item in queue if not item.get('processed', False)]

        if not unprocessed:
            log("[INFO] No unprocessed snapshots in queue")
            save_state(status="success", items_processed=0)
            return

        log(f"[INFO] Found {len(unprocessed)} snapshots to process")

        # TODO: Load vision model here if needed
        # log("[MODEL] Loading vision model...")
        # vision_model = load_vision_model()

        # Process each snapshot
        for item in unprocessed:
            snapshot_path = Path(item['snapshot_path'])

            if not snapshot_path.exists():
                log(f"[WARN] Snapshot not found: {snapshot_path}")
                item['processed'] = True
                item['error'] = "File not found"
                errors.append(f"Snapshot not found: {snapshot_path}")
                continue

            # Process with vision LLM
            analysis = process_snapshot_with_vision_llm(snapshot_path, item)

            # Update queue item
            item['processed'] = True
            item['analysis'] = analysis
            item['processed_at'] = int(time.time() * 1000)

            # Store in episodic memory with analysis
            snapshot_info = {
                **item,
                "analysis": analysis
            }

            episodic.store_episode(
                user_id="system_vision",
                username="CameraVision",
                user_message=f"Vision analysis: {item['detected_object']} {item['event_type']}",
                bot_response=json.dumps(analysis),
                hemisphere="sensory",
                salience_score=0.85  # High salience - visual evidence
            )

            items_processed += 1
            log(f"[PROGRESS] Processed {items_processed}/{len(unprocessed)}")

        # Save updated queue
        with open(SNAPSHOT_QUEUE_PATH, 'w') as f:
            json.dump(queue, f, indent=2)

        # TODO: Unload vision model if needed
        # log("[MODEL] Unloading vision model...")
        # unload_vision_model()

        duration = time.time() - start_time
        log(f"[COMPLETE] Processed {items_processed} snapshots in {duration:.2f}s")

        save_state(status="success", items_processed=items_processed, errors=errors)

    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        save_state(status="error", items_processed=items_processed, errors=[str(e)])
        sys.exit(1)

    finally:
        signal.alarm(0)  # Cancel alarm


if __name__ == "__main__":
    main()
