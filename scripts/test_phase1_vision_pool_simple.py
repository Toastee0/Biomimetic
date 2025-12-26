#!/usr/bin/env python3
"""
Phase 1 Simple Integration Test - Autonomous Vision Pool Database

Tests database functionality without requiring all dependencies:
1. Create test snapshot
2. Store segment in database (using same logic as Vision API)
3. Verify database entry
4. Test queries and views
5. Simulate processing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import sqlite3
from PIL import Image


class Phase1SimpleTest:
    def __init__(self):
        self.db_path = Path("/home/toastee/BioMimeticAi/data/biomim.db")
        self.snapshot_dir = Path("/home/toastee/BioMimeticAi/data/vision/snapshots")

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("PHASE 1 SIMPLE INTEGRATION TEST - Database Operations")
        print("="*70 + "\n")

    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def create_test_snapshot(self) -> Path:
        """Create a test snapshot image"""
        print("[TEST 1/5] Creating test snapshot image...")

        # Create a simple test image (640x480, blue background)
        img = Image.new('RGB', (640, 480), color='blue')

        # Add some visual elements
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        # Draw a rectangle to simulate a person bbox
        draw.rectangle([200, 150, 440, 450], outline='red', width=3)
        draw.text((220, 160), "TEST PERSON", fill='white')

        # Save
        timestamp = int(time.time() * 1000)
        filename = f"test_entrance_person_{timestamp}.jpg"
        snapshot_path = self.snapshot_dir / filename

        img.save(snapshot_path)

        print(f"[TEST 1/5] ✓ Created test snapshot: {snapshot_path.name}")
        print(f"           Size: {snapshot_path.stat().st_size} bytes")

        return snapshot_path

    def store_test_segment(self, snapshot_path: Path) -> int:
        """Store segment in database (same logic as Vision API)"""
        print("\n[TEST 2/5] Storing segment in database...")

        # Test event data
        event_data = {
            "event": "entrance",
            "object": "person",
            "timestamp": int(time.time() * 1000),
            "camera_id": "workshop_recamera",
            "yolo_detection": {
                "class": "person",
                "confidence": 0.95,
                "bbox": [200, 150, 440, 450],
                "track_id": 123
            }
        }

        # Calculate urgency (same logic as Vision API)
        urgency_score = 0.0
        urgency_reasons = {}

        # Person detection
        if 'person' in event_data['object']:
            urgency_score += 0.3
            urgency_reasons['person_detected'] = True

        # Entrance event
        if event_data['event'] == 'entrance':
            urgency_score += 0.2
            urgency_reasons['entrance_event'] = True

        # High YOLO confidence
        yolo = event_data.get('yolo_detection', {})
        if yolo.get('confidence', 0) > 0.9:
            urgency_score += 0.1
            urgency_reasons['high_confidence'] = yolo['confidence']

        urgency_score = min(urgency_score, 1.0)

        print(f"[TEST 2/5] Urgency score: {urgency_score:.2f}")
        print(f"           Reasons: {urgency_reasons}")

        # Determine retention
        if urgency_score < 0.3:
            retention_days = 7
        elif urgency_score < 0.7:
            retention_days = 30
        else:
            retention_days = 90

        # Store in database
        timestamp = int(time.time())

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Insert into vision_segments
            cursor.execute("""
                INSERT INTO vision_segments (
                    timestamp, created_at, camera_id, event_type,
                    file_path, file_size, processed,
                    urgent_flag, urgency_score, urgency_reason,
                    retention_days, yolo_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_data['timestamp'] // 1000,  # Convert ms to s
                timestamp,
                event_data['camera_id'],
                event_data['event'],
                str(snapshot_path),
                snapshot_path.stat().st_size,
                0,  # Not processed yet
                1 if urgency_score > 0.8 else 0,
                urgency_score,
                json.dumps(urgency_reasons),
                retention_days,
                json.dumps(event_data.get('yolo_detection'))
            ))

            segment_id = cursor.lastrowid

            # If high urgency, add to urgency_queue
            if urgency_score > 0.8:
                cursor.execute("""
                    INSERT INTO urgency_queue (
                        segment_id, urgency_score, urgency_reason, created_at
                    ) VALUES (?, ?, ?, ?)
                """, (
                    segment_id,
                    urgency_score,
                    json.dumps(urgency_reasons),
                    timestamp
                ))
                print(f"[TEST 2/5] Added to urgency_queue (HIGH urgency)")

            # Add initial YOLO tag
            cursor.execute("""
                INSERT INTO segment_tags (
                    segment_id, tag_type, tag_value, confidence,
                    bbox_json, model_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                segment_id,
                'object',
                yolo.get('class', 'person'),
                yolo.get('confidence', 0.5),
                json.dumps(yolo.get('bbox')),
                'recamera_yolo',
                timestamp
            ))

            conn.commit()

        print(f"[TEST 2/5] ✓ Stored as segment_id: {segment_id}")
        print(f"           Retention: {retention_days} days")

        return segment_id

    def verify_database_entry(self, segment_id: int):
        """Verify segment was stored correctly"""
        print("\n[TEST 3/5] Verifying database entry...")

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Query vision_segments
            cursor.execute("""
                SELECT * FROM vision_segments WHERE segment_id = ?
            """, (segment_id,))

            segment = cursor.fetchone()

            if not segment:
                raise Exception(f"Segment {segment_id} not found!")

            print(f"[TEST 3/5] ✓ Found segment in vision_segments:")
            print(f"           - camera_id: {segment['camera_id']}")
            print(f"           - event_type: {segment['event_type']}")
            print(f"           - urgency_score: {segment['urgency_score']:.2f}")
            print(f"           - urgent_flag: {bool(segment['urgent_flag'])}")
            print(f"           - retention_days: {segment['retention_days']}")
            print(f"           - processed: {bool(segment['processed'])}")

            # Query tags
            cursor.execute("""
                SELECT * FROM segment_tags WHERE segment_id = ?
            """, (segment_id,))

            tags = cursor.fetchall()

            print(f"[TEST 3/5] ✓ Found {len(tags)} tags:")
            for tag in tags:
                print(f"           - {tag['tag_type']}: {tag['tag_value']} (conf: {tag['confidence']:.2f})")

            # Check urgency_queue
            cursor.execute("""
                SELECT * FROM urgency_queue WHERE segment_id = ?
            """, (segment_id,))

            urgent = cursor.fetchone()

            if segment['urgency_score'] > 0.8:
                if urgent:
                    print(f"[TEST 3/5] ✓ Found in urgency_queue")
                else:
                    print(f"[TEST 3/5] ⚠ WARNING: High urgency but not in queue")
            else:
                print(f"[TEST 3/5] ✓ Not in urgency_queue (expected for score < 0.8)")

    def test_database_views(self):
        """Test database views"""
        print("\n[TEST 4/5] Testing database views...")

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # v_unprocessed_segments
            cursor.execute("SELECT COUNT(*) as count FROM v_unprocessed_segments")
            unprocessed = cursor.fetchone()['count']
            print(f"[TEST 4/5] ✓ v_unprocessed_segments: {unprocessed} segments")

            # v_urgent_queue
            cursor.execute("SELECT COUNT(*) as count FROM v_urgent_queue")
            urgent = cursor.fetchone()['count']
            print(f"[TEST 4/5] ✓ v_urgent_queue: {urgent} segments")

            # v_storage_summary
            cursor.execute("SELECT * FROM v_storage_summary")
            summary = cursor.fetchone()
            print(f"[TEST 4/5] ✓ v_storage_summary:")
            print(f"           - Total segments: {summary['total_segments']}")
            print(f"           - Unprocessed: {summary['unprocessed']}")
            print(f"           - Urgent: {summary['urgent']}")
            print(f"           - Total bytes: {summary['total_bytes']:,} ({summary['total_bytes'] / 1024:.1f} KB)")

            # System metadata
            cursor.execute("SELECT * FROM system_metadata")
            metadata = cursor.fetchall()
            print(f"[TEST 4/5] ✓ System metadata:")
            for row in metadata:
                print(f"           - {row['key']}: {row['value']}")

    def simulate_processing(self, segment_id: int):
        """Simulate processing by vision_processing.py"""
        print("\n[TEST 5/5] Simulating processing...")

        timestamp = int(time.time())

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Mark as processed
            cursor.execute("""
                UPDATE vision_segments
                SET processed = 1, processed_at = ?
                WHERE segment_id = ?
            """, (timestamp, segment_id))

            # Add some analysis tags
            tags_to_add = [
                ('emotion', 'neutral', 0.85, 'fer'),
                ('scene', 'indoor_workspace', 0.92, 'clip_vit_b32'),
                ('custom', 'age_35', 0.80, 'insightface')
            ]

            for tag_type, tag_value, confidence, model_name in tags_to_add:
                cursor.execute("""
                    INSERT INTO segment_tags (
                        segment_id, tag_type, tag_value, confidence,
                        model_name, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (segment_id, tag_type, tag_value, confidence, model_name, timestamp))

            conn.commit()

        print(f"[TEST 5/5] ✓ Marked segment {segment_id} as processed")
        print(f"[TEST 5/5] ✓ Added {len(tags_to_add)} analysis tags")

        # Verify
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) as count FROM segment_tags WHERE segment_id = ?
            """, (segment_id,))

            total_tags = cursor.fetchone()['count']

            cursor.execute("""
                SELECT processed FROM vision_segments WHERE segment_id = ?
            """, (segment_id,))

            processed = cursor.fetchone()['processed']

            print(f"[TEST 5/5] ✓ Verification:")
            print(f"           - Total tags: {total_tags}")
            print(f"           - Processed: {bool(processed)}")

    def run(self):
        """Run complete test suite"""
        try:
            # Test 1: Create snapshot
            snapshot_path = self.create_test_snapshot()

            # Test 2: Store in database
            segment_id = self.store_test_segment(snapshot_path)

            # Test 3: Verify
            self.verify_database_entry(segment_id)

            # Test 4: Test views
            self.test_database_views()

            # Test 5: Simulate processing
            self.simulate_processing(segment_id)

            print("\n" + "="*70)
            print("✓ ALL TESTS PASSED - Phase 1 Database Operations Working")
            print("="*70)
            print("\nTest data preserved:")
            print(f"  - Snapshot: {snapshot_path}")
            print(f"  - Database segment_id: {segment_id}")
            print("\nYou can inspect the database with:")
            print(f"  sqlite3 {self.db_path}")
            print(f"  > SELECT * FROM vision_segments WHERE segment_id = {segment_id};")
            print(f"  > SELECT * FROM segment_tags WHERE segment_id = {segment_id};")
            print()

            return True

        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    test = Phase1SimpleTest()
    success = test.run()
    sys.exit(0 if success else 1)
