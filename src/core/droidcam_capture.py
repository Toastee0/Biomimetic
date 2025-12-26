#!/usr/bin/env python3
"""DroidCam Snapshot Capture

Captures snapshots from DroidCam stream when triggered by vision events.
Uses droidcam-cli to create virtual camera device, then captures via cv2.VideoCapture(0).
"""

import subprocess
import os
import time
import socket
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import base64
import cv2
import logging

logger = logging.getLogger(__name__)


class DroidCamManager:
    """Manages DroidCam CLI connection with auto-restart capability"""

    def __init__(self, phone_ip: str = "192.168.2.39", port: int = 4747):
        """
        Initialize DroidCam manager

        Args:
            phone_ip: IP address of the phone running DroidCam
            port: Port number for DroidCam (default 4747)
        """
        self.phone_ip = phone_ip
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.polling_thread: Optional[threading.Thread] = None
        self.should_poll = False

    def start(self) -> bool:
        """
        Start DroidCam CLI connection

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running and self.process:
            logger.info("DroidCam already running, checking status...")
            if self.check_status():
                return True
            else:
                logger.warning("DroidCam process exists but not working, restarting...")
                self.stop()

        try:
            self.stop_polling()
            logger.info(f"Starting DroidCam connection to {self.phone_ip}:{self.port}...")

            # Force kill any existing droidcam-cli processes first
            subprocess.run(["pkill", "-9", "droidcam-cli"], check=False)
            time.sleep(0.5)

            # Start droidcam-cli in background
            self.process = subprocess.Popen(
                ["droidcam-cli", self.phone_ip, str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            time.sleep(0.2)

            if self.process.poll() is None:
                self.is_running = True
                logger.info(f"✓ DroidCam process started: {self.phone_ip}:{self.port}")
                time.sleep(1.5)

                if self.check_status():
                    logger.info("✓ Camera is accessible")
                else:
                    logger.warning("⚠ DroidCam process running but camera not yet accessible")
                return True
            else:
                logger.error(f"✗ DroidCam process crashed - check if app is running on {self.phone_ip}")
                self.is_running = False
                return False

        except Exception as e:
            logger.error(f"✗ Failed to start DroidCam: {e}")
            self.is_running = False
            return False

    def stop(self) -> bool:
        """Stop DroidCam CLI connection"""
        try:
            self.stop_polling()

            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=2)
                except Exception as e:
                    logger.warning(f"Failed to terminate process: {e}")
                    try:
                        self.process.kill()
                    except:
                        pass
                self.process = None

            subprocess.run(["pkill", "-9", "droidcam-cli"], check=False)
            self.is_running = False
            logger.info("DroidCam stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop DroidCam: {e}")
            return False

    def restart(self) -> bool:
        """Restart DroidCam connection"""
        logger.info("Restarting DroidCam...")
        self.stop()
        time.sleep(0.5)

        if self.is_port_reachable():
            logger.info("Camera port is reachable, connecting now")
            return self.start()
        else:
            logger.info("Camera not ready, starting polling...")
            self.start_polling()
            return True

    def check_status(self) -> bool:
        """Check if DroidCam process is running AND camera is accessible"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "droidcam-cli"],
                capture_output=True,
                text=True
            )
            process_alive = result.returncode == 0

            if not process_alive:
                self.is_running = False
                return False

            # Verify camera device is accessible
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    self.is_running = ret
                    return ret
                else:
                    cap.release()
                    self.is_running = False
                    return False
            except Exception as cam_err:
                logger.warning(f"Camera access check failed: {cam_err}")
                self.is_running = process_alive
                return process_alive

        except Exception as e:
            logger.error(f"Failed to check DroidCam status: {e}")
            self.is_running = False
            return False

    def is_port_reachable(self, timeout: float = 1.0) -> bool:
        """Check if DroidCam port is reachable on the phone"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.phone_ip, self.port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Port check failed: {e}")
            return False

    def _polling_loop(self):
        """Background thread that polls for camera availability"""
        logger.info(f"📡 DroidCam polling started - checking {self.phone_ip}:{self.port} every 5s")
        poll_count = 0

        while self.should_poll:
            poll_count += 1

            if not self.is_running:
                logger.info(f"📡 Poll #{poll_count}: Checking camera port...")

                if self.is_port_reachable():
                    logger.info(f"✓ DroidCam port {self.phone_ip}:{self.port} is reachable")
                    success = self.start()
                    if success:
                        logger.info("✓✓✓ DroidCam connected via polling!")
                        self.should_poll = False
                        break
                    else:
                        logger.warning(f"⚠ Connection attempt #{poll_count} failed")
                else:
                    logger.info(f"⏳ Poll #{poll_count}: Camera port not reachable")
            else:
                logger.info(f"✓ Poll #{poll_count}: DroidCam already running")
                self.should_poll = False
                break

            time.sleep(5)

        logger.info(f"📡 DroidCam polling stopped (ran {poll_count} polls)")

    def start_polling(self):
        """Start background polling for camera availability"""
        if self.polling_thread and self.polling_thread.is_alive():
            logger.warning("Polling already active")
            return

        self.should_poll = True
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        logger.info("Started DroidCam polling thread")

    def stop_polling(self):
        """Stop background polling"""
        self.should_poll = False
        if self.polling_thread:
            self.polling_thread.join(timeout=1)
        logger.info("Stopped DroidCam polling thread")


class DroidCamCapture:
    """Captures snapshots from DroidCam virtual camera device"""

    def __init__(self, phone_ip: str = None, port: int = None, snapshot_dir: str = None):
        """
        Initialize DroidCam capture

        Args:
            phone_ip: IP address of phone (default: 192.168.2.39)
            port: DroidCam port (default: 4747)
            snapshot_dir: Directory to save snapshots (default: data/vision/snapshots)
        """
        self.phone_ip = phone_ip or "192.168.2.39"
        self.port = port or 4747

        # Set up snapshot directory
        if snapshot_dir is None:
            self.snapshot_dir = Path(__file__).parent.parent.parent / "data" / "vision" / "snapshots"
        else:
            self.snapshot_dir = Path(snapshot_dir)

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DroidCam manager
        self.manager = DroidCamManager(phone_ip=self.phone_ip, port=self.port)

        print(f"[DROIDCAM] Initialized - Phone: {self.phone_ip}:{self.port}")
        print(f"[DROIDCAM] Snapshots will be saved to: {self.snapshot_dir}")

    def ensure_connected(self) -> bool:
        """Ensure DroidCam connection is active"""
        if not self.manager.check_status():
            print(f"[DROIDCAM] Not connected, attempting to start...")
            return self.manager.start()
        return True

    def capture_snapshot(self, filename: str = None, timeout: int = 10) -> Optional[Path]:
        """
        Capture a single frame from DroidCam virtual camera

        Args:
            filename: Output filename (default: auto-generated with timestamp)
            timeout: Maximum time to wait for capture (seconds)

        Returns:
            Path to saved image file, or None if capture failed
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"droidcam_snapshot_{timestamp}.jpg"

        output_path = self.snapshot_dir / filename

        try:
            print(f"[DROIDCAM] Capturing snapshot: {filename}")
            start_time = time.time()

            # Ensure connection
            if not self.ensure_connected():
                print(f"[DROIDCAM] ✗ Connection failed")
                return None

            # Open camera (device 0 is created by droidcam-cli)
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print(f"[DROIDCAM] ✗ Failed to open camera device")
                cap.release()
                return None

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                print(f"[DROIDCAM] ✗ Failed to read frame")
                return None

            # Save frame as JPEG
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            duration = time.time() - start_time
            file_size = output_path.stat().st_size

            print(f"[DROIDCAM] ✓ Snapshot captured in {duration:.2f}s ({file_size} bytes)")
            return output_path

        except Exception as e:
            print(f"[DROIDCAM] ✗ Unexpected error: {e}")
            return None

    def capture_snapshot_base64(self, timeout: int = 10) -> Optional[str]:
        """
        Capture snapshot and return as base64-encoded string

        Args:
            timeout: Maximum time to wait for capture (seconds)

        Returns:
            Base64-encoded JPEG image, or None if capture failed
        """
        temp_filename = f"temp_droidcam_{int(time.time() * 1000)}.jpg"
        snapshot_path = self.capture_snapshot(filename=temp_filename, timeout=timeout)

        if snapshot_path is None:
            return None

        try:
            with open(snapshot_path, 'rb') as f:
                image_data = f.read()

            base64_image = base64.b64encode(image_data).decode('utf-8')
            snapshot_path.unlink()
            return base64_image

        except Exception as e:
            print(f"[DROIDCAM] Error encoding image: {e}")
            return None

    def cleanup_old_snapshots(self, max_age_hours: int = 24, keep_count: int = 100):
        """Clean up old snapshot files to save disk space"""
        try:
            snapshot_files = sorted(
                self.snapshot_dir.glob("droidcam_snapshot_*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if len(snapshot_files) <= keep_count:
                return

            files_to_keep = set(snapshot_files[:keep_count])
            age_threshold = time.time() - (max_age_hours * 3600)

            deleted_count = 0
            deleted_bytes = 0

            for snapshot_file in snapshot_files:
                if snapshot_file in files_to_keep:
                    continue

                if snapshot_file.stat().st_mtime < age_threshold:
                    file_size = snapshot_file.stat().st_size
                    snapshot_file.unlink()
                    deleted_count += 1
                    deleted_bytes += file_size

            if deleted_count > 0:
                print(f"[DROIDCAM] Cleaned up {deleted_count} old snapshots ({deleted_bytes / 1024 / 1024:.2f} MB)")

        except Exception as e:
            print(f"[DROIDCAM] Error during cleanup: {e}")

    def test_connection(self) -> bool:
        """Test if DroidCam connection is working"""
        print(f"[DROIDCAM] Testing connection to {self.phone_ip}:{self.port}...")

        if not self.manager.start():
            print(f"[DROIDCAM] ✗ Failed to start DroidCam")
            return False

        snapshot_path = self.capture_snapshot(filename="connection_test_droidcam.jpg", timeout=15)

        if snapshot_path:
            snapshot_path.unlink()
            print(f"[DROIDCAM] ✓ Connection test successful")
            return True
        else:
            print(f"[DROIDCAM] ✗ Connection test failed")
            return False


if __name__ == "__main__":
    """Test DroidCam capture"""
    import sys

    # Allow custom IP/port from command line
    phone_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.2.39"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 4747

    capture = DroidCamCapture(phone_ip=phone_ip, port=port)

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
        print("\n[TEST] Connection test failed - check DroidCam app and network")
        sys.exit(1)
