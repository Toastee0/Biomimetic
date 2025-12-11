#!/bin/bash
# Start World Model API Server

cd /home/toastee/BioMimeticAi

echo "Starting World Model API Server..."
echo "API will be available at http://localhost:8001"
echo "WebSocket at ws://localhost:8001/ws"
echo ""

source venv/bin/activate
python -m uvicorn src.world_model.api:app --host 0.0.0.0 --port 8001 --reload
