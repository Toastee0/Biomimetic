#!/usr/bin/env python3
"""Camera Source Manager - Handles multiple camera sources for vision system

Supports switching between different camera sources (RTSP, DroidCam, etc.)
allowing the AI to shift attention between different visual inputs.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from .rtsp_capture import RTSPCapture
from .droidcam_capture import DroidCamCapture


class CameraSourceType(Enum):
    """Supported camera source types"""
    RTSP = "rtsp"
    DROIDCAM = "droidcam"


class CameraSourceManager:
    """Manages multiple camera sources and provides unified capture interface"""

    def __init__(self, config_path: str = None):
        """
        Initialize camera source manager

        Args:
            config_path: Path to camera sources config file
        """
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "camera_sources.json"

        self.config_path = Path(config_path)

        # Load configuration
        self.config = self._load_config()

        # Active camera sources
        self.sources: Dict[str, Any] = {}

        # Current active source
        self.active_source_name: Optional[str] = None
        self.active_source = None

        # Initialize configured sources
        self._initialize_sources()

        print(f"[CAMERA MANAGER] Initialized with {len(self.sources)} sources")
        print(f"[CAMERA MANAGER] Active source: {self.active_source_name}")

    def _load_config(self) -> Dict:
        """Load camera sources configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"[CAMERA MANAGER] Loaded config from {self.config_path}")
                return config
            else:
                # Return default configuration
                print(f"[CAMERA MANAGER] Config not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            print(f"[CAMERA MANAGER] Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default camera sources configuration"""
        return {
            "active_source": "rtsp_recamera",
            "sources": {
                "rtsp_recamera": {
                    "type": "rtsp",
                    "enabled": True,
                    "name": "reCamera (Front Door)",
                    "rtsp_url": "rtsp://192.168.2.140:8554/stream",
                    "snapshot_dir": "data/vision/snapshots"
                },
                "droidcam_phone": {
                    "type": "droidcam",
                    "enabled": True,
                    "name": "DroidCam Phone",
                    "phone_ip": "192.168.2.39",
                    "port": 4747,
                    "snapshot_dir": "data/vision/snapshots"
                }
            }
        }

    def _initialize_sources(self):
        """Initialize all enabled camera sources"""
        for source_name, source_config in self.config.get("sources", {}).items():
            if not source_config.get("enabled", True):
                print(f"[CAMERA MANAGER] Skipping disabled source: {source_name}")
                continue

            source_type = source_config.get("type")

            try:
                if source_type == "rtsp":
                    self.sources[source_name] = RTSPCapture(
                        rtsp_url=source_config.get("rtsp_url"),
                        snapshot_dir=source_config.get("snapshot_dir")
                    )
                    print(f"[CAMERA MANAGER] ✓ Initialized RTSP source: {source_name}")

                elif source_type == "droidcam":
                    self.sources[source_name] = DroidCamCapture(
                        phone_ip=source_config.get("phone_ip"),
                        port=source_config.get("port"),
                        snapshot_dir=source_config.get("snapshot_dir")
                    )
                    print(f"[CAMERA MANAGER] ✓ Initialized DroidCam source: {source_name}")

                else:
                    print(f"[CAMERA MANAGER] ✗ Unknown source type '{source_type}' for {source_name}")

            except Exception as e:
                print(f"[CAMERA MANAGER] ✗ Failed to initialize {source_name}: {e}")

        # Set active source
        active_source_name = self.config.get("active_source")
        if active_source_name and active_source_name in self.sources:
            self.active_source_name = active_source_name
            self.active_source = self.sources[active_source_name]
        elif self.sources:
            # Default to first available source
            self.active_source_name = list(self.sources.keys())[0]
            self.active_source = self.sources[self.active_source_name]
        else:
            print(f"[CAMERA MANAGER] ⚠ No camera sources available!")

    def switch_source(self, source_name: str) -> bool:
        """
        Switch to a different camera source

        Args:
            source_name: Name of the source to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        if source_name not in self.sources:
            print(f"[CAMERA MANAGER] ✗ Unknown source: {source_name}")
            print(f"[CAMERA MANAGER] Available sources: {list(self.sources.keys())}")
            return False

        self.active_source_name = source_name
        self.active_source = self.sources[source_name]

        source_config = self.config["sources"][source_name]
        print(f"[CAMERA MANAGER] ✓ Switched to: {source_config.get('name', source_name)}")

        return True

    def capture_snapshot(self, filename: str = None, timeout: int = 10, source: str = None) -> Optional[Path]:
        """
        Capture snapshot from specified or active camera source

        Args:
            filename: Output filename (default: auto-generated)
            timeout: Capture timeout in seconds
            source: Specific source to use (default: active source)

        Returns:
            Path to captured snapshot, or None if failed
        """
        # Determine which source to use
        if source:
            if source not in self.sources:
                print(f"[CAMERA MANAGER] ✗ Unknown source: {source}")
                return None
            capture_source = self.sources[source]
            source_name = source
        else:
            if not self.active_source:
                print(f"[CAMERA MANAGER] ✗ No active source available")
                return None
            capture_source = self.active_source
            source_name = self.active_source_name

        print(f"[CAMERA MANAGER] Capturing from: {source_name}")
        return capture_source.capture_snapshot(filename=filename, timeout=timeout)

    def capture_snapshot_base64(self, timeout: int = 10, source: str = None) -> Optional[str]:
        """
        Capture snapshot as base64 from specified or active camera source

        Args:
            timeout: Capture timeout in seconds
            source: Specific source to use (default: active source)

        Returns:
            Base64-encoded image, or None if failed
        """
        if source:
            if source not in self.sources:
                print(f"[CAMERA MANAGER] ✗ Unknown source: {source}")
                return None
            capture_source = self.sources[source]
        else:
            if not self.active_source:
                print(f"[CAMERA MANAGER] ✗ No active source available")
                return None
            capture_source = self.active_source

        return capture_source.capture_snapshot_base64(timeout=timeout)

    def get_source_info(self, source_name: str = None) -> Optional[Dict]:
        """
        Get information about a camera source

        Args:
            source_name: Name of source (default: active source)

        Returns:
            Dictionary with source information
        """
        if source_name is None:
            source_name = self.active_source_name

        if source_name not in self.config["sources"]:
            return None

        source_config = self.config["sources"][source_name]

        return {
            "name": source_name,
            "display_name": source_config.get("name", source_name),
            "type": source_config.get("type"),
            "enabled": source_config.get("enabled", True),
            "is_active": source_name == self.active_source_name,
            "config": source_config
        }

    def list_sources(self) -> Dict[str, Dict]:
        """
        List all available camera sources

        Returns:
            Dictionary mapping source names to their info
        """
        sources_info = {}
        for source_name in self.config.get("sources", {}).keys():
            info = self.get_source_info(source_name)
            if info:
                sources_info[source_name] = info
        return sources_info

    def test_connection(self, source_name: str = None) -> bool:
        """
        Test connection for specified or active source

        Args:
            source_name: Name of source to test (default: active source)

        Returns:
            True if connection successful, False otherwise
        """
        if source_name is None:
            source_name = self.active_source_name

        if source_name not in self.sources:
            print(f"[CAMERA MANAGER] ✗ Unknown source: {source_name}")
            return False

        print(f"[CAMERA MANAGER] Testing connection: {source_name}")
        return self.sources[source_name].test_connection()


if __name__ == "__main__":
    """Test camera source manager"""
    import sys

    manager = CameraSourceManager()

    print("\n=== Available Sources ===")
    for source_name, info in manager.list_sources().items():
        active_marker = "✓ ACTIVE" if info["is_active"] else ""
        print(f"{source_name}: {info['display_name']} ({info['type']}) {active_marker}")

    print("\n=== Testing Active Source ===")
    manager.test_connection()

    # Test switching sources if argument provided
    if len(sys.argv) > 1:
        new_source = sys.argv[1]
        print(f"\n=== Switching to {new_source} ===")
        if manager.switch_source(new_source):
            manager.test_connection()

            print(f"\n=== Capturing test snapshot ===")
            snapshot_path = manager.capture_snapshot()
            if snapshot_path:
                print(f"✓ Snapshot saved: {snapshot_path}")