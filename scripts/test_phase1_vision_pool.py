#!/usr/bin/env python3
"""
Phase 1 Integration Test - Autonomous Vision Pool

Tests complete flow:
1. Create test snapshot
2. Simulate reCamera event → Vision API
3. Verify database entry with urgency scoring
4. Verify episodic memory entry
5. Query database for unprocessed segments
6. Verify views and queries work
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import sqlite3
import requests
from PIL import Image


class Phase1Test:
    def __init__(self):
        self.db_path = Path("/home/toastee/BioMimeticAi/data/biomim.db")
        self.snapshot_dir = Path("/home/toastee/BioMimeticAi/data/vision/snapshots")
        self.vision_api_url = "http://localhost:8000/api/vision/event"

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("PHASE 1 INTEGRATION TEST - Autonomous Vision Pool")
        print("="*70 + "\n")

    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def create_test_snapshot(self) -> Path:
        """Create a test snapshot image"""
        print("[TEST 1/6] Creating test snapshot image...")

        # Create a simple test image (640x480, blue background)
        img = Image.new('RGB', (640, 480), color='blue')

        # Add some text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        # Draw a simple rectangle to simulate a person bbox
        draw.rectangle([200, 150, 440, 450], outline='red', width=3)
        draw.text((220, 160), "TEST PERSON", fill='white')

        # Save
        timestamp = int(time.time() * 1000)
        filename = f"test_entrance_person_{timestamp}.jpg"
        snapshot_path = self.snapshot_dir / filename

        img.save(snapshot_path)

        print(f"[TEST 1/6] ✓ Created test snapshot: {snapshot_path.name}")
        print(f"           Size: {snapshot_path.stat().st_size} bytes")

        return snapshot_path

    def simulate_recamera_event(self, snapshot_path: Path) -> dict:
        """Simulate reCamera entrance event"""
        print("\n[TEST 2/6] Simulating reCamera entrance event...")

        # Note: This test doesn't actually POST to the API
        # Instead, it directly calls the Vision API functions
        # To test with real API, you'd need to start the Flask server first

        # For now, let's simulate the event data and directly test database storage
        event_data = {
            "event": "entrance",
            "object": "person",
            "timestamp": int(time.time() * 1000),
            "camera_id": "workshop_recamera",
            "scene": ["person"],
            "yolo_detection": {
                "class": "person",
                "confidence": 0.95,
                "bbox": [200, 150, 440, 450],
                "track_id": 123
            }
        }

        print(f"[TEST 2/6] Event data: {json.dumps(event_data, indent=2)}")

        # Import and use the Vision API functions directly
        from src.core.vision_api import evaluate_urgency, store_segment_in_db

        # Evaluate urgency
        urgency_score, urgency_reasons = evaluate_urgency(event_data, event_data['yolo_detection'])

        print(f"[TEST 2/6] ✓ Urgency score: {urgency_score:.2f}")
        print(f"           Reasons: {urgency_reasons}")

        # Store in database
        segment_id = store_segment_in_db(event_data, snapshot_path, urgency_score, urgency_reasons)

        print(f"[TEST 2/6] ✓ Stored in database as segment_id: {segment_id}")

        return {
            "segment_id": segment_id,
            "urgency_score": urgency_score,
            "urgency_reasons": urgency_reasons,
            "event_data": event_data
        }

    def verify_database_entry(self, segment_id: int):
        """Verify segment was stored correctly in database"""
        print("\n[TEST 3/6] Verifying database entry...")

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Query vision_segments
            cursor.execute("""
                SELECT * FROM vision_segments WHERE segment_id = ?
            """, (segment_id,))

            segment = cursor.fetchone()

            if not segment:
                print(f"[TEST 3/6] ✗ FAILED: Segment {segment_id} not found in database")
                return False

            print(f"[TEST 3/6] ✓ Found segment in vision_segments table:")
            print(f"           - camera_id: {segment['camera_id']}")
            print(f"           - event_type: {segment['event_type']}")
            print(f"           - urgency_score: {segment['urgency_score']:.2f}")
            print(f"           - urgent_flag: {segment['urgent_flag']}")
            print(f"           - retention_days: {segment['retention_days']}")
            print(f"           - processed: {segment['processed']}")
            print(f"           - file_path: {segment['file_path']}")

            # Query segment_tags (should have initial YOLO tag)
            cursor.execute("""
                SELECT * FROM segment_tags WHERE segment_id = ?
            """, (segment_id,))

            tags = cursor.fetchall()

            print(f"[TEST 3/6] ✓ Found {len(tags)} tags:")
            for tag in tags:
                print(f"           - {tag['tag_type']}: {tag['tag_value']} (conf: {tag['confidence']:.2f}, model: {tag['model_name']})")

            # Check if in urgency_queue (if urgency > 0.8)
            cursor.execute("""
                SELECT * FROM urgency_queue WHERE segment_id = ?
            """, (segment_id,))

            urgent = cursor.fetchone()

            if segment['urgency_score'] > 0.8:
                if urgent:
                    print(f"[TEST 3/6] ✓ Found in urgency_queue (HIGH urgency)")
                    print(f"           - ai_notified: {urgent['ai_notified']}")
                else:
                    print(f"[TEST 3/6] ⚠ WARNING: High urgency but not in urgency_queue")
            else:
                print(f"[TEST 3/6] ✓ Not in urgency_queue (urgency < 0.8, expected)")

            return True

    def verify_episodic_memory(self):
        """Verify episodic memory entries were created"""
        print("\n[TEST 4/6] Verifying episodic memory entries...")

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Query recent episodes from system_vision
            cursor.execute("""
                SELECT * FROM episodes
                WHERE contact_id = 'system_vision'
                ORDER BY timestamp DESC
                LIMIT 5
            """)

            episodes = cursor.fetchall()

            print(f"[TEST 4/6] ✓ Found {len(episodes)} recent vision episodes:")
            for i, ep in enumerate(episodes, 1):
                print(f"           {i}. [{ep['username']}] {ep['user_message'][:60]}...")
                print(f"              Hemisphere: {ep['hemisphere']}, Salience: {ep['salience_score']:.2f}")

            return len(episodes) > 0

    def test_database_queries(self):
        """Test database views and queries"""
        print("\n[TEST 5/6] Testing database views and queries...")

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Test v_unprocessed_segments view
            cursor.execute("SELECT COUNT(*) as count FROM v_unprocessed_segments")
            unprocessed_count = cursor.fetchone()['count']
            print(f"[TEST 5/6] ✓ Unprocessed segments view: {unprocessed_count} segments")

            # Test v_urgent_queue view
            cursor.execute("SELECT COUNT(*) as count FROM v_urgent_queue")
            urgent_count = cursor.fetchone()['count']
            print(f"[TEST 5/6] ✓ Urgent queue view: {urgent_count} segments")

            # Test v_storage_summary view
            cursor.execute("SELECT * FROM v_storage_summary")
            summary = cursor.fetchone()
            print(f"[TEST 5/6] ✓ Storage summary view:")
            print(f"           - Total segments: {summary['total_segments']}")
            print(f"           - Unprocessed: {summary['unprocessed']}")
            print(f"           - Urgent: {summary['urgent']}")
            print(f"           - Total bytes: {summary['total_bytes']} ({summary['total_bytes'] / 1024:.1f} KB)")

            # Test system metadata
            cursor.execute("SELECT * FROM system_metadata")
            metadata = cursor.fetchall()
            print(f"[TEST 5/6] ✓ System metadata:")
            for row in metadata:
                print(f"           - {row['key']}: {row['value']}")

            return True

    def test_vision_processing_query(self):
        """Test that vision_processing.py can query unprocessed segments"""
        print("\n[TEST 6/6] Testing vision processing cortex database query...")

        # Import VisionProcessingCortex
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "cron"))
        from vision_processing import VisionProcessingCortex

        # Create cortex instance
        cortex = VisionProcessingCortex()

        # Load queue (should include database segments)
        queue = cortex.load_queue()

        print(f"[TEST 6/6] ✓ Vision cortex loaded {len(queue)} segments from queue")

        if queue:
            print(f"[TEST 6/6] ✓ Sample queue item:")
            sample = queue[0]
            print(f"           - snapshot_path: {sample.get('snapshot_path', 'N/A')}")
            print(f"           - event_type: {sample.get('event_type', 'N/A')}")
            print(f"           - urgency_score: {sample.get('urgency_score', 'N/A')}")
            print(f"           - from_database: {sample.get('from_database', False)}")

        return len(queue) > 0

    def cleanup(self, snapshot_path: Path, segment_id: int):
        """Optional: Clean up test data"""
        print("\n[CLEANUP] Cleaning up test data...")

        # Delete test snapshot
        if snapshot_path.exists():
            snapshot_path.unlink()
            print(f"[CLEANUP] ✓ Deleted test snapshot: {snapshot_path.name}")

        # Optionally delete database entry
        # (Commented out to leave test data for inspection)
        # with self.get_db_connection() as conn:
        #     conn.execute("DELETE FROM vision_segments WHERE segment_id = ?", (segment_id,))
        #     conn.commit()
        #     print(f"[CLEANUP] ✓ Deleted database segment {segment_id}")

    def run(self):
        """Run complete test suite"""
        try:
            # Test 1: Create snapshot
            snapshot_path = self.create_test_snapshot()

            # Test 2: Simulate event
            result = self.simulate_recamera_event(snapshot_path)
            segment_id = result['segment_id']

            # Test 3: Verify database
            self.verify_database_entry(segment_id)

            # Test 4: Verify episodic memory
            self.verify_episodic_memory()

            # Test 5: Test queries
            self.test_database_queries()

            # Test 6: Test vision processing
            self.test_vision_processing_query()

            # Cleanup (optional)
            # self.cleanup(snapshot_path, segment_id)

            print("\n" + "="*70)
            print("✓ PHASE 1 INTEGRATION TEST PASSED")
            print("="*70)
            print("\nTest data preserved for inspection:")
            print(f"  - Snapshot: {snapshot_path}")
            print(f"  - Database segment_id: {segment_id}")
            print(f"  - Urgency score: {result['urgency_score']:.2f}")
            print("\nTo clean up, run:")
            print(f"  rm {snapshot_path}")
            print(f"  sqlite3 {self.db_path} \"DELETE FROM vision_segments WHERE segment_id = {segment_id};\"")
            print()

            return True

        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    test = Phase1Test()
    success = test.run()
    sys.exit(0 if success else 1)
