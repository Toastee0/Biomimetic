#!/bin/bash
# Test camera vision system end-to-end

cd /home/toastee/BioMimeticAi
source venv/bin/activate

echo "========================================="
echo "Camera Vision System Test"
echo "========================================="
echo ""

# Check if ffmpeg is installed
echo "1. Checking dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    echo "   ✗ ffmpeg not found"
    echo "   Install with: sudo apt install ffmpeg"
    exit 1
else
    echo "   ✓ ffmpeg found"
fi

# Test RTSP connection
echo ""
echo "2. Testing RTSP stream connection..."
python3 src/core/rtsp_capture.py
if [ $? -eq 0 ]; then
    echo "   ✓ RTSP connection successful"
else
    echo "   ✗ RTSP connection failed"
    echo "   Check that reCamera is accessible at 192.168.2.140"
    exit 1
fi

# Test SSH connection
echo ""
echo "3. Testing SSH connection to reCamera..."
python3 scripts/camera_ssh.py "echo 'SSH test successful'"
if [ $? -eq 0 ]; then
    echo "   ✓ SSH connection successful"
else
    echo "   ✗ SSH connection failed"
    echo "   Check SSH credentials and network"
    exit 1
fi

echo ""
echo "========================================="
echo "All tests passed! ✓"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Start vision API: python src/core/vision_api.py"
echo "2. Deploy entrance/exit tracker to reCamera Node-RED"
echo "3. Monitor snapshots: watch -n 1 'ls -lh data/vision/snapshots/'"
echo ""
