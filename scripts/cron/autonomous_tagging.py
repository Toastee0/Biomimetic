#!/usr/bin/env python3
"""
Autonomous Tagging Cortex
Periodically processes untagged vision segments through the tagging pipeline

Cron Schedule: Every 10 minutes
RTOS Timeout: 300 seconds (5 minutes)
"""

import os
import sys
import signal
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.tagging_pipeline import TaggingPipeline

# RTOS Timeout Configuration
TIMEOUT_SECONDS = 300  # 5 minute timeout

# Paths
STATE_FILE = Path("data/cortex_state/autonomous_tagging.json")
LOG_FILE = Path("logs/autonomous_tagging.log")

# Ensure directories exist
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(message: str):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)

    # Append to log file
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    log(f"[TIMEOUT] Process exceeded {TIMEOUT_SECONDS}s limit")
    save_state({
        "last_run": int(time.time()),
        "status": "timeout",
        "duration_seconds": TIMEOUT_SECONDS,
        "segments_processed": 0,
        "errors": ["Process timeout"]
    })
    sys.exit(124)  # Timeout exit code


def save_state(state: dict):
    """Save cortex state to JSON file"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"[ERROR] Failed to save state: {e}")


def main():
    """Main autonomous tagging cortex loop"""
    log("="*70)
    log("AUTONOMOUS TAGGING CORTEX - Starting")
    log("="*70)

    # Set up RTOS timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    start_time = time.time()
    state = {
        "last_run": int(start_time),
        "status": "running",
        "duration_seconds": 0,
        "segments_processed": 0,
        "yolo_tags": 0,
        "clip_embeddings": 0,
        "errors": []
    }

    try:
        # Initialize tagging pipeline
        log("[INIT] Initializing TaggingPipeline...")
        pipeline = TaggingPipeline()

        # Process batch (50 segments at a time)
        batch_size = 50
        log(f"[PROCESS] Processing batch of {batch_size} segments...")

        stats = pipeline.process_batch(batch_size=batch_size)

        # Update state with results
        state["segments_processed"] = stats["total_segments"]
        state["yolo_tags"] = stats["yolo_tags"]
        state["clip_embeddings"] = stats["clip_embeddings"]

        if stats["errors"] > 0:
            state["errors"].append(f"{stats['errors']} processing errors occurred")

        # Calculate duration
        duration = time.time() - start_time
        state["duration_seconds"] = round(duration, 2)
        state["status"] = "success"

        # Log results
        log(f"[SUCCESS] Processed {stats['total_segments']} segments in {duration:.2f}s")
        log(f"  - YOLO tags: {stats['yolo_tags']}")
        log(f"  - CLIP embeddings: {stats['clip_embeddings']}")
        log(f"  - Errors: {stats['errors']}")

        # Save state
        save_state(state)

        # Cancel alarm
        signal.alarm(0)

        log("="*70)
        log("AUTONOMOUS TAGGING CORTEX - Complete")
        log("="*70)

        return 0

    except Exception as e:
        # Handle errors
        duration = time.time() - start_time
        state["duration_seconds"] = round(duration, 2)
        state["status"] = "error"
        state["errors"].append(str(e))

        log(f"[ERROR] Autonomous tagging cortex failed: {e}")

        # Print traceback for debugging
        import traceback
        traceback.print_exc()

        # Save error state
        save_state(state)

        # Cancel alarm
        signal.alarm(0)

        log("="*70)
        log("AUTONOMOUS TAGGING CORTEX - Failed")
        log("="*70)

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
