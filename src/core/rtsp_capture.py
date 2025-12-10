#!/usr/bin/env python3
"""RTSP Stream Snapshot Capture

Captures snapshots from reCamera RTSP stream when triggered by entrance/exit events.
Uses ffmpeg for efficient single-frame capture without maintaining persistent connection.
"""

import subprocess
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import base64


class RTSPCapture:
    """Captures snapshots from RTSP stream"""

    def __init__(self, rtsp_url: str = None, snapshot_dir: str = None):
        """
        Initialize RTSP capture

        Args:
            rtsp_url: RTSP stream URL (default: rtsp://192.168.2.140:8554/stream)
            snapshot_dir: Directory to save snapshots (default: data/vision/snapshots)
        """
        self.rtsp_url = rtsp_url or "rtsp://192.168.2.140:8554/stream"

        # Set up snapshot directory
        if snapshot_dir is None:
            self.snapshot_dir = Path(__file__).parent.parent.parent / "data" / "vision" / "snapshots"
        else:
            self.snapshot_dir = Path(snapshot_dir)

        # Create directory if it doesn't exist
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RTSP] Initialized - Stream: {self.rtsp_url}")
        print(f"[RTSP] Snapshots will be saved to: {self.snapshot_dir}")

    def capture_snapshot(self, filename: str = None, timeout: int = 10) -> Optional[Path]:
        """
        Capture a single frame from RTSP stream using ffmpeg

        Args:
            filename: Output filename (default: auto-generated with timestamp)
            timeout: Maximum time to wait for capture (seconds)

        Returns:
            Path to saved image file, or None if capture failed
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"snapshot_{timestamp}.jpg"

        output_path = self.snapshot_dir / filename

        # ffmpeg command to capture single frame
        # -rtsp_transport tcp: Use TCP for more reliable connection
        # -i {rtsp_url}: Input stream
        # -frames:v 1: Capture exactly 1 frame
        # -q:v 2: JPEG quality (2 = high quality)
        # -y: Overwrite output file if exists
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-frames:v", "1",
            "-q:v", "2",
            "-y",
            str(output_path)
        ]

        try:
            print(f"[RTSP] Capturing snapshot: {filename}")
            start_time = time.time()

            # Run ffmpeg with timeout
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False
            )

            duration = time.time() - start_time

            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size
                print(f"[RTSP] ✓ Snapshot captured in {duration:.2f}s ({file_size} bytes)")
                return output_path
            else:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"[RTSP] ✗ Capture failed (exit code {result.returncode})")
                print(f"[RTSP] Error: {error_msg[-500:]}")  # Last 500 chars of error
                return None

        except subprocess.TimeoutExpired:
            print(f"[RTSP] ✗ Capture timed out after {timeout}s")
            return None
        except FileNotFoundError:
            print(f"[RTSP] ✗ ffmpeg not found - install with: sudo apt install ffmpeg")
            return None
        except Exception as e:
            print(f"[RTSP] ✗ Unexpected error: {e}")
            return None

    def capture_snapshot_base64(self, timeout: int = 10) -> Optional[str]:
        """
        Capture snapshot and return as base64-encoded string (for LLM vision APIs)

        Args:
            timeout: Maximum time to wait for capture (seconds)

        Returns:
            Base64-encoded JPEG image, or None if capture failed
        """
        # Capture to temp file
        temp_filename = f"temp_{int(time.time() * 1000)}.jpg"
        snapshot_path = self.capture_snapshot(filename=temp_filename, timeout=timeout)

        if snapshot_path is None:
            return None

        try:
            # Read and encode
            with open(snapshot_path, 'rb') as f:
                image_data = f.read()

            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Clean up temp file
            snapshot_path.unlink()

            return base64_image

        except Exception as e:
            print(f"[RTSP] Error encoding image: {e}")
            return None

    def cleanup_old_snapshots(self, max_age_hours: int = 24, keep_count: int = 100):
        """
        Clean up old snapshot files to save disk space

        Args:
            max_age_hours: Delete files older than this many hours
            keep_count: Always keep at least this many recent files
        """
        try:
            # Get all snapshot files sorted by modification time (newest first)
            snapshot_files = sorted(
                self.snapshot_dir.glob("snapshot_*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if len(snapshot_files) <= keep_count:
                return  # Not enough files to clean up

            # Keep the most recent keep_count files
            files_to_keep = set(snapshot_files[:keep_count])

            # Calculate age threshold
            age_threshold = time.time() - (max_age_hours * 3600)

            deleted_count = 0
            deleted_bytes = 0

            for snapshot_file in snapshot_files:
                if snapshot_file in files_to_keep:
                    continue

                # Delete if older than threshold
                if snapshot_file.stat().st_mtime < age_threshold:
                    file_size = snapshot_file.stat().st_size
                    snapshot_file.unlink()
                    deleted_count += 1
                    deleted_bytes += file_size

            if deleted_count > 0:
                print(f"[RTSP] Cleaned up {deleted_count} old snapshots ({deleted_bytes / 1024 / 1024:.2f} MB)")

        except Exception as e:
            print(f"[RTSP] Error during cleanup: {e}")

    def test_connection(self) -> bool:
        """
        Test if RTSP stream is accessible

        Returns:
            True if stream is accessible, False otherwise
        """
        print(f"[RTSP] Testing connection to {self.rtsp_url}...")
        snapshot_path = self.capture_snapshot(filename="connection_test.jpg", timeout=15)

        if snapshot_path:
            # Clean up test file
            snapshot_path.unlink()
            print(f"[RTSP] ✓ Connection test successful")
            return True
        else:
            print(f"[RTSP] ✗ Connection test failed")
            return False


if __name__ == "__main__":
    """Test RTSP capture"""
    import sys

    # Allow custom RTSP URL from command line
    rtsp_url = sys.argv[1] if len(sys.argv) > 1 else None

    capture = RTSPCapture(rtsp_url=rtsp_url)

    # Test connection
    if capture.test_connection():
        print("\n[TEST] Capturing 3 test snapshots...")
        for i in range(3):
            snapshot_path = capture.capture_snapshot()
            if snapshot_path:
                print(f"[TEST] {i+1}/3 - Saved to {snapshot_path}")
                time.sleep(1)
            else:
                print(f"[TEST] {i+1}/3 - Failed")
                break

        print("\n[TEST] Testing base64 encoding...")
        base64_data = capture.capture_snapshot_base64()
        if base64_data:
            print(f"[TEST] ✓ Base64 encoded ({len(base64_data)} chars)")
        else:
            print(f"[TEST] ✗ Base64 encoding failed")
    else:
        print("\n[TEST] Connection test failed - check RTSP URL and network")
        sys.exit(1)
