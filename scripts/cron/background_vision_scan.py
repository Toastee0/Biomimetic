#!/usr/bin/env python3
"""
Background Vision Scan - Safety net for YOLO misses
Captures general snapshot every 20 seconds (runs every 1 minute with internal loop)
"""

import signal
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.rtsp_capture import RTSPCapture
from src.core.model_manager import ModelManager
from src.memory.episodic import EpisodicMemory

TIMEOUT_SECONDS = 55  # 55 seconds (5s buffer)
SCAN_INTERVAL = 20  # Capture every 20 seconds
STATE_PATH = Path(__file__).parent.parent.parent / "data" / "cortex_state" / "background_vision_scan.json"
LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "background_vision_scan.log"
SCAN_DIR = Path(__file__).parent.parent.parent / "data" / "vision" / "background_scans"

# Ensure directories exist
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
SCAN_DIR.mkdir(parents=True, exist_ok=True)


def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_PATH, 'a') as f:
        f.write(log_message + "\n")


def timeout_handler(signum, frame):
    """Handle timeout"""
    log(f"[TIMEOUT] Process exceeded {TIMEOUT_SECONDS}s limit")
    save_state(status="timeout", items_processed=0, errors=["Timeout"])
    sys.exit(124)


def save_state(status, items_processed, errors=None):
    """Save cortex state"""
    state = {
        "last_run": int(time.time()),
        "status": status,
        "items_processed": items_processed,
        "errors": errors or [],
        "next_scheduled": int(time.time()) + 60  # 1 minute
    }
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def analyze_scene_clip(model_manager: ModelManager, snapshot_path: Path) -> dict:
    """
    Analyze scene with CLIP to detect what YOLO might have missed
    
    Args:
        model_manager: Model manager instance
        snapshot_path: Path to snapshot
        
    Returns:
        Analysis results dict
    """
    try:
        # Load CLIP
        if not model_manager.load_clip():
            log("[SCAN] CLIP not available")
            return {"error": "CLIP not available"}
        
        # Load image
        image = Image.open(snapshot_path)
        
        # Define queries for things YOLO might miss
        queries = [
            "empty room",
            "person in the scene",
            "multiple people",
            "pet animal",
            "car or vehicle",
            "unusual activity",
            "normal everyday scene",
            "dark or night scene",
            "bright daylight scene"
        ]
        
        # Analyze with CLIP
        scores = model_manager.analyze_scene_clip(image, queries)
        
        if not scores:
            return {"error": "CLIP analysis failed"}
        
        # Get top match
        top_match = max(scores.items(), key=lambda x: x[1])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "snapshot_path": str(snapshot_path),
            "top_scene": top_match[0],
            "confidence": top_match[1],
            "all_scores": scores
        }
        
    except Exception as e:
        log(f"[SCAN ERROR] Analysis failed: {e}")
        return {"error": str(e)}


def cleanup_old_scans(max_age_hours: int = 24):
    """Clean up old background scan images"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        deleted_count = 0
        for scan_file in SCAN_DIR.glob("background_scan_*.jpg"):
            file_age = current_time - scan_file.stat().st_mtime
            
            if file_age > max_age_seconds:
                scan_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            log(f"[CLEANUP] Deleted {deleted_count} old scans")
            
    except Exception as e:
        log(f"[CLEANUP ERROR] {e}")


def main():
    """Main background scan loop"""
    start_time = time.time()
    log("[START] Background Vision Scan Cortex")

    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        rtsp = RTSPCapture()
        model_manager = ModelManager()
        episodic = EpisodicMemory()
        scans_captured = 0
        errors = []

        # Run for ~50 seconds (2 scans at 20s interval)
        runtime_limit = 50
        last_scan = 0

        while time.time() - start_time < runtime_limit:
            current_time = time.time() - start_time

            # Check if it's time for next scan
            if current_time - last_scan >= SCAN_INTERVAL:
                log(f"[SCAN] Capturing background scan #{scans_captured + 1}")

                # Capture snapshot
                timestamp = int(time.time() * 1000)
                filename = f"background_scan_{timestamp}.jpg"
                snapshot_path = rtsp.capture_snapshot(filename=filename, timeout=10)

                if snapshot_path:
                    log(f"[SCAN] ✓ Captured: {snapshot_path.name}")

                    # Analyze with CLIP
                    analysis = analyze_scene_clip(model_manager, snapshot_path)

                    if "error" not in analysis:
                        top_scene = analysis.get("top_scene", "unknown")
                        confidence = analysis.get("confidence", 0.0)
                        
                        log(f"[SCAN] Scene: {top_scene} (confidence: {confidence:.2f})")

                        # Store in episodic memory if significant
                        salience = 0.4 if confidence > 0.5 else 0.3
                        
                        episodic.store_episode(
                            user_id="system_vision",
                            username="BackgroundScan",
                            user_message=f"Background scan: {top_scene} (confidence: {confidence:.2f})",
                            bot_response=json.dumps(analysis),
                            hemisphere="sensory",
                            salience_score=salience
                        )
                    else:
                        log(f"[SCAN] Analysis error: {analysis.get('error')}")

                    scans_captured += 1
                else:
                    log(f"[SCAN] ✗ Failed to capture scan")
                    errors.append("Failed to capture snapshot")

                last_scan = current_time

            # Sleep briefly to avoid busy-wait
            time.sleep(1)

        # Cleanup old scans
        cleanup_old_scans()

        # Unload models
        model_manager.unload_all()

        duration = time.time() - start_time
        log(f"[COMPLETE] Captured {scans_captured} background scans in {duration:.2f}s")

        save_state(status="success", items_processed=scans_captured, errors=errors)

    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        save_state(status="error", items_processed=0, errors=[str(e)])
        sys.exit(1)

    finally:
        signal.alarm(0)  # Cancel alarm


if __name__ == "__main__":
    main()
            current_time = time.time()

            # Check if it's time for next scan
            if current_time - last_scan >= SCAN_INTERVAL:
                log(f"[SCAN] Capturing background scan #{scans_captured + 1}")

                # Capture snapshot
                filename = f"background_scan_{int(current_time * 1000)}.jpg"
                snapshot_path = rtsp.capture_snapshot(filename=filename, timeout=10)

                if snapshot_path:
                    # Queue for processing
                    queue_scan_for_processing(snapshot_path)

                    # Store basic event in episodic memory
                    episodic.store_episode(
                        user_id="system_vision",
                        username="BackgroundScan",
                        user_message=f"Background vision scan captured (queued for processing)",
                        bot_response=json.dumps({"snapshot_path": str(snapshot_path)}),
                        hemisphere="sensory",
                        salience_score=0.3  # Low salience - just routine scan
                    )

                    scans_captured += 1
                    log(f"[SCAN] ✓ Scan captured and queued")
                else:
                    log(f"[SCAN] ✗ Failed to capture scan")
                    errors.append("Failed to capture snapshot")

                last_scan = current_time

            # Sleep briefly to avoid busy-wait
            time.sleep(1)

        duration = time.time() - start_time
        log(f"[COMPLETE] Captured {scans_captured} background scans in {duration:.2f}s")

        save_state(status="success", items_processed=scans_captured, errors=errors)

    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        save_state(status="error", items_processed=0, errors=[str(e)])
        sys.exit(1)

    finally:
        signal.alarm(0)  # Cancel alarm


if __name__ == "__main__":
    main()
