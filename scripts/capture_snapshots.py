#!/usr/bin/env python3
"""Capture snapshots from all available cameras"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime

# Import camera capture class
from src.core.rtsp_capture import RTSPCapture

def capture_from_recamera(camera_num, rtsp_url):
    """Capture snapshot from reCamera via RTSP"""
    print(f"\n[CAPTURE] Capturing from reCamera {camera_num}...")
    print(f"          RTSP: {rtsp_url}")

    output_dir = Path("/home/toastee/BioMimeticAi/data/vision/snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recamera{camera_num}_snapshot_{timestamp}.jpg"

    try:
        # Create RTSPCapture instance
        capture = RTSPCapture(rtsp_url=rtsp_url, snapshot_dir=str(output_dir))

        # Capture snapshot
        result = capture.capture_snapshot(filename=filename, timeout=10)

        if result and result.exists():
            size_kb = result.stat().st_size / 1024
            print(f"[CAPTURE] ✓ reCamera {camera_num} snapshot saved: {result}")
            print(f"          Size: {size_kb:.1f} KB")
            return result
        else:
            print(f"[CAPTURE] ✗ Failed to capture from reCamera {camera_num}")
            return None

    except Exception as e:
        print(f"[CAPTURE ERROR] reCamera {camera_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

def capture_from_droidcam():
    """Capture snapshot from DroidCam via HTTP"""
    print("\n[CAPTURE] Capturing from DroidCam Phone...")

    phone_ip = "192.168.2.39"
    port = 4747
    output_dir = Path("/home/toastee/BioMimeticAi/data/vision/snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"droidcam_snapshot_{timestamp}.jpg"
    output_path = output_dir / filename

    try:
        # Try to import DroidCam capture
        from src.core.droidcam_capture import DroidCamCapture

        droidcam = DroidCamCapture(phone_ip, port)
        result = droidcam.capture_snapshot(str(output_path))

        if result and output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"[CAPTURE] ✓ DroidCam snapshot saved: {output_path}")
            print(f"          Size: {size_kb:.1f} KB")
            return output_path
        else:
            print(f"[CAPTURE] ✗ Failed to capture from DroidCam")
            return None

    except ImportError:
        print(f"[CAPTURE] ⚠ DroidCam capture module not available (missing cv2)")
        print(f"          Trying HTTP snapshot instead...")

        # Fallback: Direct HTTP request
        try:
            import requests
            url = f"http://{phone_ip}:{port}/photo.jpg"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                size_kb = output_path.stat().st_size / 1024
                print(f"[CAPTURE] ✓ DroidCam snapshot saved (HTTP): {output_path}")
                print(f"          Size: {size_kb:.1f} KB")
                return output_path
            else:
                print(f"[CAPTURE] ✗ HTTP request failed: {response.status_code}")
                return None

        except Exception as e:
            print(f"[CAPTURE ERROR] DroidCam HTTP fallback: {e}")
            return None

    except Exception as e:
        print(f"[CAPTURE ERROR] DroidCam: {e}")
        return None

def main():
    print("="*70)
    print("CAMERA SNAPSHOT CAPTURE - All reCameras")
    print("="*70)

    snapshots = {}

    # Capture from reCamera 1 (192.168.2.140)
    recamera1_url = "rtsp://admin:admin@192.168.2.140:554/live"
    recamera1_snapshot = capture_from_recamera(1, recamera1_url)
    if recamera1_snapshot:
        snapshots['recamera_1'] = recamera1_snapshot

    # Small delay between captures
    time.sleep(1)

    # Capture from reCamera 2 (192.168.2.143)
    recamera2_url = "rtsp://admin:admin@192.168.2.143:554/live"
    recamera2_snapshot = capture_from_recamera(2, recamera2_url)
    if recamera2_snapshot:
        snapshots['recamera_2'] = recamera2_snapshot

    # Summary
    print("\n" + "="*70)
    print(f"CAPTURE COMPLETE - {len(snapshots)} snapshots saved")
    print("="*70)

    if snapshots:
        print("\nSnapshots:")
        for camera, path in snapshots.items():
            print(f"  {camera}: {path}")
        print("\nYou can view these images with:")
        for camera, path in snapshots.items():
            print(f"  eog {path}  # (or your preferred image viewer)")
    else:
        print("\n⚠ No snapshots captured - check camera connectivity")

    print()

if __name__ == "__main__":
    main()
