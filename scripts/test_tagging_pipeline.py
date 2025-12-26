#!/usr/bin/env python3
"""
Test the autonomous tagging pipeline with real camera snapshots
"""

import sys
import sqlite3
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.tagging_pipeline import TaggingPipeline


def get_db_connection():
    """Get database connection with WAL mode"""
    conn = sqlite3.connect("data/biomim.db", timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def store_test_segments():
    """Store existing snapshots as unprocessed segments for testing"""
    snapshot_dir = Path("data/vision/snapshots")
    snapshots = list(snapshot_dir.glob("recamera*_snapshot_*.jpg"))

    print(f"\n=== Storing {len(snapshots)} test segments ===\n")

    segment_ids = []

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for snapshot in snapshots:
            # Extract camera ID from filename
            camera_id = "recamera_1" if "recamera1" in snapshot.name else "recamera_2"
            timestamp = int(time.time())

            # Check if already exists
            cursor.execute("""
                SELECT segment_id FROM vision_segments
                WHERE file_path = ?
            """, (str(snapshot),))

            existing = cursor.fetchone()

            if existing:
                segment_id = existing[0]
                # Mark as unprocessed for testing
                cursor.execute("""
                    UPDATE vision_segments
                    SET processed = 0
                    WHERE segment_id = ?
                """, (segment_id,))
                print(f"Reset existing segment {segment_id}: {snapshot.name}")
            else:
                # Insert new segment
                cursor.execute("""
                    INSERT INTO vision_segments
                    (timestamp, camera_id, file_path, processed,
                     urgent_flag, urgency_score, retention_days, created_at)
                    VALUES (?, ?, ?, 0, 0, 0.5, 30, ?)
                """, (timestamp, camera_id, str(snapshot), timestamp))

                segment_id = cursor.lastrowid
                print(f"Created segment {segment_id}: {snapshot.name}")

            segment_ids.append(segment_id)

        conn.commit()

    print(f"\n✓ {len(segment_ids)} test segments ready")
    return segment_ids


def view_results(segment_ids):
    """View tagging results for test segments"""
    print("\n=== Tagging Results ===\n")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for segment_id in segment_ids:
            # Get segment info
            cursor.execute("""
                SELECT file_path, camera_id, processed
                FROM vision_segments
                WHERE segment_id = ?
            """, (segment_id,))

            row = cursor.fetchone()
            file_path = Path(row[0]).name
            camera_id = row[1]
            processed = row[2]

            print(f"Segment {segment_id} ({camera_id}): {file_path}")
            print(f"  Processed: {'Yes' if processed else 'No'}")

            # Get YOLO object tags
            cursor.execute("""
                SELECT tag_value, confidence, bbox_json
                FROM segment_tags
                WHERE segment_id = ? AND tag_type = 'object'
                ORDER BY confidence DESC
            """, (segment_id,))

            objects = cursor.fetchall()
            if objects:
                print(f"  YOLO Objects ({len(objects)}):")
                for obj, conf, bbox in objects[:5]:  # Top 5
                    print(f"    - {obj} ({conf:.2%})")
            else:
                print("  YOLO Objects: None detected")

            # Get CLIP scene tags
            cursor.execute("""
                SELECT tag_value, confidence
                FROM segment_tags
                WHERE segment_id = ? AND tag_type = 'scene'
                ORDER BY confidence DESC
            """, (segment_id,))

            scenes = cursor.fetchall()
            if scenes:
                print(f"  CLIP Scenes:")
                for scene, conf in scenes:
                    print(f"    - {scene} ({conf:.2%})")
            else:
                print("  CLIP Scenes: None")

            # Check CLIP embedding
            cursor.execute("""
                SELECT embedding_id FROM scene_embeddings
                WHERE segment_id = ?
            """, (segment_id,))

            has_embedding = cursor.fetchone() is not None
            print(f"  CLIP Embedding: {'Yes' if has_embedding else 'No'}")

            print()


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("TAGGING PIPELINE TEST")
    print("="*70)

    # Store test segments
    segment_ids = store_test_segments()

    if not segment_ids:
        print("\n✗ No test segments found")
        print("  Run scripts/capture_snapshots.py first")
        return 1

    # Initialize pipeline
    print("\n=== Initializing TaggingPipeline ===\n")
    pipeline = TaggingPipeline()

    # Process batch
    print("\n=== Processing Batch ===\n")
    stats = pipeline.process_batch(batch_size=10)

    # Show statistics
    print("\n=== Processing Statistics ===\n")
    print(f"Total segments: {stats['total_segments']}")
    print(f"YOLO tags: {stats['yolo_tags']}")
    print(f"CLIP embeddings: {stats['clip_embeddings']}")
    print(f"Errors: {stats['errors']}")
    print(f"Processing time: {stats['processing_time_sec']:.2f}s")

    # View detailed results
    view_results(segment_ids)

    print("="*70)
    print("TEST COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
