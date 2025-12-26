#!/usr/bin/env python3
"""
Autonomous Tagging Pipeline
Sequentially loads vision models, processes segments, and stores tags in database
"""

import os
import sys
import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaggingPipeline:
    """
    Autonomous tagging pipeline that processes vision segments sequentially

    Pipeline stages:
    1. YOLO (~50MB) - Object detection
    2. TSM (~150MB) - Activity classification
    3. CLIP (~350MB) - Scene embeddings
    4. PANNs (~150MB) - Audio event detection (if audio available)

    Total VRAM budget: 700MB (loads one model at a time)
    """

    def __init__(
        self,
        db_path: str = "data/biomim.db",
        vram_buffer_mb: int = 200
    ):
        """
        Initialize tagging pipeline

        Args:
            db_path: Path to SQLite database
            vram_buffer_mb: VRAM buffer to maintain (MB)
        """
        self.db_path = Path(db_path)
        self.vram_buffer_mb = vram_buffer_mb
        self.model_manager = ModelManager()

        # Verify database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        logger.info("TaggingPipeline initialized")

    def get_db_connection(self):
        """Get database connection with WAL mode"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def get_unprocessed_segments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get unprocessed segments from database

        Args:
            limit: Maximum segments to retrieve

        Returns:
            List of segment dicts with metadata
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT segment_id, file_path, camera_id, timestamp,
                       urgency_score, yolo_metadata
                FROM vision_segments
                WHERE processed = 0
                ORDER BY urgency_score DESC, timestamp ASC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()

            segments = []
            for row in rows:
                segments.append({
                    "segment_id": row[0],
                    "file_path": row[1],
                    "camera_id": row[2],
                    "timestamp": row[3],
                    "urgency_score": row[4],
                    "yolo_metadata": row[5]
                })

            logger.info(f"Retrieved {len(segments)} unprocessed segments")
            return segments

    def store_tag(
        self,
        segment_id: int,
        tag_type: str,
        tag_value: str,
        confidence: float,
        bbox_json: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Store a tag in the database

        Args:
            segment_id: Segment ID
            tag_type: Type of tag ('object', 'activity', 'scene', 'audio')
            tag_value: Tag value (e.g., 'person', 'walking', 'workshop')
            confidence: Confidence score (0.0-1.0)
            bbox_json: Bounding box as JSON string
            model_name: Name of model that generated tag
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            created_at = int(time.time())
            cursor.execute("""
                INSERT INTO segment_tags
                (segment_id, tag_type, tag_value, confidence, bbox_json, model_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (segment_id, tag_type, tag_value, confidence, bbox_json, model_name, created_at))
            conn.commit()

    def store_clip_embedding(self, segment_id: int, embedding: np.ndarray):
        """
        Store CLIP embedding in database

        Args:
            segment_id: Segment ID
            embedding: CLIP embedding vector (512-dim)
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Convert numpy array to bytes
            embedding_bytes = embedding.tobytes()
            embedding_dim = len(embedding)
            created_at = int(time.time())

            cursor.execute("""
                INSERT OR REPLACE INTO scene_embeddings
                (segment_id, embedding, embedding_dim, model_name, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (segment_id, embedding_bytes, embedding_dim, "clip-vit-b32", created_at))
            conn.commit()

    def mark_segment_processed(self, segment_id: int):
        """Mark segment as processed in database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE vision_segments
                SET processed = 1
                WHERE segment_id = ?
            """, (segment_id,))
            conn.commit()

    def process_segment_yolo(self, segment: Dict[str, Any]) -> int:
        """
        Process segment with YOLO object detection

        Args:
            segment: Segment metadata dict

        Returns:
            Number of tags added
        """
        segment_id = segment["segment_id"]
        file_path = Path(segment["file_path"])

        if not file_path.exists():
            logger.warning(f"Segment file not found: {file_path}")
            return 0

        try:
            # Load image
            image = Image.open(file_path)
            image_np = np.array(image)

            # Detect objects
            detections = self.model_manager.detect_objects_yolo(image_np, conf_threshold=0.25)

            if not detections:
                logger.info(f"No objects detected in segment {segment_id}")
                return 0

            # Store detections as tags
            tag_count = 0
            for detection in detections:
                bbox_json = str(detection["bbox"])  # Convert list to string

                self.store_tag(
                    segment_id=segment_id,
                    tag_type="object",
                    tag_value=detection["class"],
                    confidence=detection["confidence"],
                    bbox_json=bbox_json,
                    model_name="yolov8n"
                )
                tag_count += 1

            logger.info(f"Added {tag_count} YOLO tags to segment {segment_id}")
            return tag_count

        except Exception as e:
            logger.error(f"YOLO processing failed for segment {segment_id}: {e}")
            return 0

    def process_segment_clip(self, segment: Dict[str, Any]) -> bool:
        """
        Process segment with CLIP for scene embedding

        Args:
            segment: Segment metadata dict

        Returns:
            True if successful
        """
        segment_id = segment["segment_id"]
        file_path = Path(segment["file_path"])

        if not file_path.exists():
            logger.warning(f"Segment file not found: {file_path}")
            return False

        try:
            # Load image
            image = Image.open(file_path)

            # Get embedding
            embedding = self.model_manager.get_clip_embedding(image)

            if embedding is None:
                logger.warning(f"Failed to get CLIP embedding for segment {segment_id}")
                return False

            # Store embedding
            self.store_clip_embedding(segment_id, embedding)

            # Also store scene classification tags
            scene_queries = [
                "workshop or garage",
                "office or workspace",
                "living room",
                "kitchen",
                "bedroom",
                "outdoor scene",
                "empty room",
                "person working",
                "person sitting",
                "person standing"
            ]

            scene_scores = self.model_manager.analyze_scene_clip(image, scene_queries)

            if scene_scores:
                # Store top 3 scene tags
                sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
                for scene, score in sorted_scenes[:3]:
                    if score > 0.1:  # Only store if score > 10%
                        self.store_tag(
                            segment_id=segment_id,
                            tag_type="scene",
                            tag_value=scene,
                            confidence=float(score),
                            model_name="clip-vit-b32"
                        )

            logger.info(f"Added CLIP embedding and scene tags to segment {segment_id}")
            return True

        except Exception as e:
            logger.error(f"CLIP processing failed for segment {segment_id}: {e}")
            return False

    def process_batch(self, batch_size: int = 50) -> Dict[str, int]:
        """
        Process a batch of unprocessed segments through full pipeline

        Args:
            batch_size: Number of segments to process

        Returns:
            Dict with processing statistics
        """
        stats = {
            "total_segments": 0,
            "yolo_tags": 0,
            "clip_embeddings": 0,
            "errors": 0,
            "processing_time_sec": 0
        }

        start_time = time.time()

        # Get unprocessed segments
        segments = self.get_unprocessed_segments(limit=batch_size)
        stats["total_segments"] = len(segments)

        if len(segments) == 0:
            logger.info("No unprocessed segments found")
            return stats

        logger.info(f"Processing {len(segments)} segments through tagging pipeline")

        # Stage 1: YOLO Object Detection
        logger.info("Stage 1: Loading YOLO for object detection...")
        if self.model_manager.load_yolo():
            for segment in segments:
                tag_count = self.process_segment_yolo(segment)
                stats["yolo_tags"] += tag_count

            # Unload YOLO before loading next model
            self.model_manager.unload_model("yolo")
            logger.info("Stage 1 complete, YOLO unloaded")
        else:
            logger.error("Failed to load YOLO, skipping object detection stage")
            stats["errors"] += 1

        # Stage 2: CLIP Scene Understanding & Embeddings
        logger.info("Stage 2: Loading CLIP for scene understanding...")
        if self.model_manager.load_clip():
            for segment in segments:
                if self.process_segment_clip(segment):
                    stats["clip_embeddings"] += 1

            # Unload CLIP
            self.model_manager.unload_model("clip")
            logger.info("Stage 2 complete, CLIP unloaded")
        else:
            logger.error("Failed to load CLIP, skipping scene analysis stage")
            stats["errors"] += 1

        # Mark all segments as processed
        for segment in segments:
            self.mark_segment_processed(segment["segment_id"])

        stats["processing_time_sec"] = time.time() - start_time

        logger.info(f"Batch processing complete: {stats}")
        return stats


def main():
    """Test the tagging pipeline"""
    print("\n=== Tagging Pipeline Test ===\n")

    # Initialize pipeline
    pipeline = TaggingPipeline()

    # Process a small batch
    stats = pipeline.process_batch(batch_size=10)

    print(f"\nProcessing Statistics:")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  YOLO tags: {stats['yolo_tags']}")
    print(f"  CLIP embeddings: {stats['clip_embeddings']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Processing time: {stats['processing_time_sec']:.2f}s")

    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    main()
