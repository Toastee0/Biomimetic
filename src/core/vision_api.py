#!/usr/bin/env python3
"""Vision Event API - Receives entrance/exit events from camera"""

from flask import Flask, request, jsonify
import json
import time
from datetime import datetime
import sys
import os
from threading import Thread
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.episodic import EpisodicMemory
from core.camera_source_manager import CameraSourceManager
from pathlib import Path

app = Flask(__name__)
episodic = EpisodicMemory()
camera_manager = CameraSourceManager()

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "biomim.db"

# Store recent vision events in memory (could be extended to database)
recent_events = []
MAX_RECENT_EVENTS = 100

# Snapshot processing queue (lightweight - just paths)
# Vision cortex will process these on schedule
SNAPSHOT_QUEUE_PATH = Path(__file__).parent.parent.parent / "data" / "vision" / "snapshot_queue.json"
SNAPSHOT_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    """Get database connection with WAL mode and timeout"""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def evaluate_urgency(event_data, yolo_detection=None):
    """
    Evaluate urgency of event using autonomous heuristics

    Args:
        event_data: Event data from camera
        yolo_detection: Optional YOLO detection metadata

    Returns:
        (urgency_score, reason_dict)

    Urgency scoring:
        HIGH (> 0.8): Push to AI immediately
        MEDIUM (0.3 - 0.7): Priority queue for batch processing
        LOW (< 0.3): Stay in pool for lazy processing
    """
    score = 0.0
    reasons = {}

    # Person detection
    obj_name = event_data.get('object', '').lower()
    if 'person' in obj_name:
        score += 0.3
        reasons['person_detected'] = True

        # YOLO confidence boost
        if yolo_detection and yolo_detection.get('confidence', 0) > 0.9:
            score += 0.1
            reasons['high_confidence'] = yolo_detection['confidence']

    # Entrance events (more important than exits)
    if event_data.get('event') == 'entrance':
        score += 0.2
        reasons['entrance_event'] = True

    # Multiple objects in scene (unusual activity)
    scene = event_data.get('scene', [])
    if len(scene) > 2:
        score += 0.1
        reasons['multiple_objects'] = len(scene)

    # Time-based urgency (night time = more suspicious)
    current_hour = datetime.now().hour
    if current_hour < 6 or current_hour > 22:  # 10pm - 6am
        score += 0.15
        reasons['unusual_time'] = current_hour

    return min(score, 1.0), reasons


def store_segment_in_db(event_data, snapshot_path, urgency_score, urgency_reasons):
    """
    Store vision segment in database for autonomous tagging pipeline

    Args:
        event_data: Event data from camera
        snapshot_path: Path to captured snapshot
        urgency_score: Calculated urgency score
        urgency_reasons: Dictionary of urgency factors
    """
    try:
        timestamp = int(time.time())

        # Determine retention based on urgency
        if urgency_score < 0.3:
            retention_days = 7
        elif urgency_score < 0.7:
            retention_days = 30
        else:
            retention_days = 90

        with get_db_connection() as conn:
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
                event_data.get('timestamp', timestamp * 1000) // 1000,  # Convert ms to s
                timestamp,
                event_data.get('camera_id', 'workshop_recamera'),
                event_data.get('event', 'entrance'),
                str(snapshot_path),
                snapshot_path.stat().st_size if snapshot_path.exists() else 0,
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
                print(f"[URGENCY] High-urgency segment added to queue (score: {urgency_score:.2f})")

            # If YOLO detection provided, create initial object tag
            yolo = event_data.get('yolo_detection')
            if yolo:
                cursor.execute("""
                    INSERT INTO segment_tags (
                        segment_id, tag_type, tag_value, confidence,
                        bbox_json, model_name, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    segment_id,
                    'object',
                    yolo.get('class', event_data.get('object')),
                    yolo.get('confidence', 0.5),
                    json.dumps(yolo.get('bbox')),
                    'recamera_yolo',
                    timestamp
                ))

            conn.commit()

            print(f"[DATABASE] Stored segment {segment_id} (urgency: {urgency_score:.2f}, retention: {retention_days}d)")
            return segment_id

    except Exception as e:
        print(f"[DATABASE ERROR] Failed to store segment: {e}")
        return None


def push_urgent_event_to_ai(segment_id, event_data, urgency_score, urgency_reasons):
    """
    Push high-urgency event to AI for immediate attention

    This will eventually integrate with Discord notifications and
    the AI batch review system.

    Args:
        segment_id: Database segment ID
        event_data: Event data from camera
        urgency_score: Urgency score (> 0.8)
        urgency_reasons: Dictionary of urgency factors
    """
    try:
        # For now, just create high-salience episodic memory entry
        # TODO: Add Discord notification integration
        # TODO: Add HTTP webhook to AI review system

        reason_str = ", ".join([f"{k}: {v}" for k, v in urgency_reasons.items()])

        episodic.store_episode(
            user_id="system_vision",
            username="VisionUrgent",
            user_message=f"🚨 URGENT: {event_data.get('object')} {event_data.get('event')} (score: {urgency_score:.2f})",
            bot_response=f"Urgency reasons: {reason_str}\nSegment ID: {segment_id}",
            hemisphere="cognitive",
            salience_score=urgency_score
        )

        print(f"[PUSH] Urgent event pushed to AI (segment_id: {segment_id}, score: {urgency_score:.2f})")

        # Mark as notified in database
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE vision_segments
                SET ai_notified = 1
                WHERE segment_id = ?
            """, (segment_id,))

            conn.execute("""
                UPDATE urgency_queue
                SET ai_notified = 1, notified_at = ?
                WHERE segment_id = ?
            """, (int(time.time()), segment_id))

            conn.commit()

    except Exception as e:
        print(f"[PUSH ERROR] Failed to push urgent event: {e}")


def queue_snapshot_for_processing(event_data, snapshot_path):
    """
    Queue snapshot for later processing by vision cortex
    NOW WITH: Urgency detection and database storage

    Args:
        event_data: Event data from camera (includes YOLO metadata)
        snapshot_path: Path to captured snapshot

    Note: Vision cortex will process queue on schedule (not immediately)
    This is lightweight - just writes to queue file + database
    """
    try:
        print(f"[VISION QUEUE] Adding snapshot to processing queue: {snapshot_path.name}")
        print(f"[VISION QUEUE] Event: {event_data['event']} - {event_data['object']}")

        # Evaluate urgency using autonomous heuristics
        yolo_detection = event_data.get('yolo_detection')
        urgency_score, urgency_reasons = evaluate_urgency(event_data, yolo_detection)
        print(f"[URGENCY] Score: {urgency_score:.2f} - Reasons: {urgency_reasons}")

        # Store in database
        segment_id = store_segment_in_db(event_data, snapshot_path, urgency_score, urgency_reasons)

        # Load existing queue (legacy for compatibility with existing cortex)
        if SNAPSHOT_QUEUE_PATH.exists():
            with open(SNAPSHOT_QUEUE_PATH, 'r') as f:
                queue = json.load(f)
        else:
            queue = []

        # Add to queue with YOLO metadata + urgency info
        queue_entry = {
            "segment_id": segment_id,  # NEW: Database reference
            "snapshot_path": str(snapshot_path),
            "event_type": event_data['event'],
            "detected_object": event_data['object'],
            "scene": event_data.get('scene', []),
            "timestamp": event_data['timestamp'],
            "queued_at": int(time.time() * 1000),
            "processed": False,
            "yolo_detection": event_data.get('yolo_detection'),  # Include YOLO metadata
            "urgency_score": urgency_score,  # NEW: Urgency info
            "urgency_reasons": urgency_reasons  # NEW: Urgency factors
        }

        # Log YOLO metadata if present
        if queue_entry["yolo_detection"]:
            yolo = queue_entry["yolo_detection"]
            print(f"[VISION QUEUE] YOLO: class={yolo.get('class')}, conf={yolo.get('confidence'):.2f}, bbox={yolo.get('bbox')}")

        queue.append(queue_entry)

        # Save queue
        with open(SNAPSHOT_QUEUE_PATH, 'w') as f:
            json.dump(queue, f, indent=2)

        print(f"[VISION QUEUE] ✓ Queued ({len(queue)} total in queue)")
        print(f"[VISION QUEUE] Vision cortex will process on next scheduled run")

        # Push to AI if high urgency
        if urgency_score > 0.8:
            push_urgent_event_to_ai(segment_id, event_data, urgency_score, urgency_reasons)

        # Store basic event in episodic memory (without analysis)
        episodic.store_episode(
            user_id="system_vision",
            username="CameraSystem",
            user_message=f"Snapshot captured: {event_data['object']} {event_data['event']} (urgency: {urgency_score:.2f})",
            bot_response=json.dumps(queue_entry),
            hemisphere="sensory",
            salience_score=min(0.6 + (urgency_score * 0.2), 0.9)  # Scale with urgency
        )

    except Exception as e:
        print(f"[VISION QUEUE ERROR] {e}")


def trigger_snapshot_capture(event_data):
    """
    Trigger snapshot capture in background thread

    Args:
        event_data: Event data from camera
    """
    def capture_task():
        try:
            # Capture snapshot with event-specific filename
            timestamp = event_data['timestamp']
            obj_name = event_data['object'].replace(' ', '_')
            event_type = event_data['event']
            filename = f"{event_type}_{obj_name}_{timestamp}.jpg"

            print(f"[VISION] Triggering snapshot capture for {event_type}: {obj_name}")
            snapshot_path = camera_manager.capture_snapshot(filename=filename, timeout=10)

            if snapshot_path:
                # Queue snapshot for later processing by vision cortex
                queue_snapshot_for_processing(event_data, snapshot_path)
            else:
                print(f"[VISION] ✗ Failed to capture snapshot for {event_type}")

        except Exception as e:
            print(f"[VISION ERROR] Snapshot capture failed: {e}")

    # Run in background thread so we don't block the API response
    thread = Thread(target=capture_task, daemon=True)
    thread.start()


@app.route('/api/vision/event', methods=['POST'])
def receive_vision_event():
    """
    Receive entrance/exit events from camera system

    Expected payload:
    {
        "event": "entrance" | "exit",
        "object": "person" | "car" | etc.,
        "timestamp": 1234567890,
        "scene": ["person", "car"],
        "yolo_detection": {  // Optional YOLO metadata
            "class": "person",
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2],
            "track_id": 123
        }
    }
    """
    try:
        event_data = request.json

        if not event_data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ['event', 'object', 'timestamp']
        for field in required_fields:
            if field not in event_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Add server receive time
        event_data['received_at'] = int(time.time() * 1000)
        event_data['server_timestamp'] = datetime.now().isoformat()

        # Store event
        recent_events.append(event_data)
        if len(recent_events) > MAX_RECENT_EVENTS:
            recent_events.pop(0)

        # Create episodic memory entry for significant events
        event_type = event_data['event']
        obj_name = event_data['object']
        scene = event_data.get('scene', [])

        if event_type == 'entrance':
            message = f"Vision: {obj_name} entered the scene"
            if len(scene) > 1:
                message += f" (scene now has: {', '.join(scene)})"

            # IMPORTANT: Trigger snapshot capture for entrance events
            trigger_snapshot_capture(event_data)

        elif event_type == 'exit':
            duration = event_data.get('duration', 0) / 1000  # Convert to seconds
            message = f"Vision: {obj_name} left the scene (was present for {duration:.1f}s)"
            if scene:
                message += f" (scene now has: {', '.join(scene)})"
        else:
            message = f"Vision: Unknown event type {event_type} for {obj_name}"

        # Store in episodic memory with system user
        episodic.store_episode(
            user_id="system_vision",
            username="CameraSystem",
            user_message=message,
            bot_response="Event recorded",
            hemisphere="sensory",
            salience_score=0.7  # Vision events are moderately salient
        )

        print(f"[VISION EVENT] {message}")

        return jsonify({
            "status": "success",
            "message": "Event received and stored",
            "event_id": len(recent_events) - 1
        }), 200

    except Exception as e:
        print(f"[VISION ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vision/events', methods=['GET'])
def get_recent_events():
    """Get recent vision events"""
    limit = request.args.get('limit', 50, type=int)
    event_type = request.args.get('type', None)  # 'entrance' or 'exit'

    filtered_events = recent_events
    if event_type:
        filtered_events = [e for e in recent_events if e.get('event') == event_type]

    return jsonify({
        "events": filtered_events[-limit:],
        "total": len(filtered_events)
    }), 200


@app.route('/api/vision/status', methods=['GET'])
def get_status():
    """Get vision system status"""
    if recent_events:
        last_event = recent_events[-1]
        last_event_time = datetime.fromtimestamp(last_event['timestamp'] / 1000)
        last_event_ago = (datetime.now() - last_event_time).total_seconds()
    else:
        last_event = None
        last_event_ago = None

    # Get active camera source info
    active_source_info = camera_manager.get_source_info()

    return jsonify({
        "status": "online",
        "recent_event_count": len(recent_events),
        "last_event": last_event,
        "last_event_seconds_ago": last_event_ago,
        "active_camera_source": active_source_info
    }), 200


@app.route('/api/vision/sources', methods=['GET'])
def list_camera_sources():
    """List all available camera sources"""
    try:
        sources = camera_manager.list_sources()
        return jsonify({
            "status": "success",
            "sources": sources,
            "active_source": camera_manager.active_source_name
        }), 200
    except Exception as e:
        print(f"[VISION ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vision/sources/<source_name>', methods=['POST'])
def switch_camera_source(source_name):
    """Switch to a different camera source"""
    try:
        success = camera_manager.switch_source(source_name)
        if success:
            source_info = camera_manager.get_source_info(source_name)
            return jsonify({
                "status": "success",
                "message": f"Switched to camera source: {source_name}",
                "active_source": source_info
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": f"Failed to switch to source: {source_name}",
                "available_sources": list(camera_manager.sources.keys())
            }), 400
    except Exception as e:
        print(f"[VISION ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vision/sources/<source_name>/test', methods=['POST'])
def test_camera_source(source_name):
    """Test connection for a camera source"""
    try:
        success = camera_manager.test_connection(source_name)
        return jsonify({
            "status": "success" if success else "failed",
            "source": source_name,
            "connection_ok": success
        }), 200 if success else 500
    except Exception as e:
        print(f"[VISION ERROR] {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Vision Event API on port 8000...")
    print("Endpoints:")
    print("  POST /api/vision/event - Receive entrance/exit events")
    print("  GET  /api/vision/events - Get recent events")
    print("  GET  /api/vision/status - Get system status")
    print("  GET  /api/vision/sources - List available camera sources")
    print("  POST /api/vision/sources/<name> - Switch to camera source")
    print("  POST /api/vision/sources/<name>/test - Test camera source")
    print(f"\nActive camera source: {camera_manager.active_source_name}")
    app.run(host='0.0.0.0', port=8000, debug=True)
