#!/usr/bin/env python3
"""Vision Event API - Receives entrance/exit events from camera"""

from flask import Flask, request, jsonify
import json
import time
from datetime import datetime
import sys
import os
from threading import Thread

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.episodic import EpisodicMemory
from core.rtsp_capture import RTSPCapture
from pathlib import Path

app = Flask(__name__)
episodic = EpisodicMemory()
rtsp_capture = RTSPCapture()

# Store recent vision events in memory (could be extended to database)
recent_events = []
MAX_RECENT_EVENTS = 100

# Snapshot processing queue (lightweight - just paths)
# Vision cortex will process these on schedule
SNAPSHOT_QUEUE_PATH = Path(__file__).parent.parent.parent / "data" / "vision" / "snapshot_queue.json"
SNAPSHOT_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)


def queue_snapshot_for_processing(event_data, snapshot_path):
    """
    Queue snapshot for later processing by vision cortex

    Args:
        event_data: Event data from camera (includes YOLO metadata)
        snapshot_path: Path to captured snapshot

    Note: Vision cortex will process queue on schedule (not immediately)
    This is lightweight - just writes to queue file
    """
    try:
        print(f"[VISION QUEUE] Adding snapshot to processing queue: {snapshot_path.name}")
        print(f"[VISION QUEUE] Event: {event_data['event']} - {event_data['object']}")

        # Load existing queue
        if SNAPSHOT_QUEUE_PATH.exists():
            with open(SNAPSHOT_QUEUE_PATH, 'r') as f:
                queue = json.load(f)
        else:
            queue = []

        # Add to queue with YOLO metadata
        queue_entry = {
            "snapshot_path": str(snapshot_path),
            "event_type": event_data['event'],
            "detected_object": event_data['object'],
            "scene": event_data.get('scene', []),
            "timestamp": event_data['timestamp'],
            "queued_at": int(time.time() * 1000),
            "processed": False,
            "yolo_detection": event_data.get('yolo_detection')  # Include YOLO metadata
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

        # Store basic event in episodic memory (without analysis)
        episodic.store_episode(
            user_id="system_vision",
            username="CameraSystem",
            user_message=f"Snapshot captured: {event_data['object']} {event_data['event']} (queued for processing)",
            bot_response=json.dumps(queue_entry),
            hemisphere="sensory",
            salience_score=0.6  # Lower until processed
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
            snapshot_path = rtsp_capture.capture_snapshot(filename=filename, timeout=10)

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

    return jsonify({
        "status": "online",
        "recent_event_count": len(recent_events),
        "last_event": last_event,
        "last_event_seconds_ago": last_event_ago
    }), 200


if __name__ == "__main__":
    print("Starting Vision Event API on port 8000...")
    print("Endpoints:")
    print("  POST /api/vision/event - Receive entrance/exit events")
    print("  GET  /api/vision/events - Get recent events")
    print("  GET  /api/vision/status - Get system status")
    app.run(host='0.0.0.0', port=8000, debug=True)
