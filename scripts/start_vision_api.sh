#!/bin/bash
# Start Vision Event API server

cd /home/toastee/BioMimeticAi
source venv/bin/activate

echo "Starting Vision Event API..."
echo "Endpoints:"
echo "  POST http://192.168.2.137:8000/api/vision/event"
echo "  GET  http://192.168.2.137:8000/api/vision/events"
echo "  GET  http://192.168.2.137:8000/api/vision/status"
echo ""
echo "This API will:"
echo "  - Receive entrance/exit events from reCamera"
echo "  - Capture RTSP snapshots on entrance events"
echo "  - Process snapshots with vision LLM"
echo "  - Store events in episodic memory"
echo ""

python3 src/core/vision_api.py
