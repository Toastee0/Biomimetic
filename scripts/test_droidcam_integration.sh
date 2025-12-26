#!/bin/bash
# Test script for DroidCam integration

echo "=== DroidCam Integration Test ==="
echo ""

# Activate venv
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

echo "1. Testing DroidCamCapture module..."
python3 src/core/droidcam_capture.py
CAPTURE_RESULT=$?

echo ""
echo "2. Testing CameraSourceManager..."
python3 src/core/camera_source_manager.py
MANAGER_RESULT=$?

echo ""
echo "=== Test Summary ==="
if [ $CAPTURE_RESULT -eq 0 ]; then
    echo "✓ DroidCamCapture: PASS"
else
    echo "✗ DroidCamCapture: FAIL (exit code: $CAPTURE_RESULT)"
fi

if [ $MANAGER_RESULT -eq 0 ]; then
    echo "✓ CameraSourceManager: PASS"
else
    echo "✗ CameraSourceManager: FAIL (exit code: $MANAGER_RESULT)"
fi

echo ""
echo "Note: Tests will fail if DroidCam app is not running on phone"
echo "or if phone is not reachable at configured IP address."

exit $(($CAPTURE_RESULT + $MANAGER_RESULT))
