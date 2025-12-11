#!/bin/bash
# Start World Model with OpenArena 3D viewer

cd /home/toastee/BioMimeticAi

echo "=== Starting World Model 3D Viewer (OpenArena) ==="
echo ""

# Start World Model API in background
echo "Starting World Model API..."
source venv/bin/activate
python -m uvicorn src.world_model.api:app --host 0.0.0.0 --port 8001 &
WM_PID=$!
echo "World Model API started (PID: $WM_PID)"

# Wait for API to start
sleep 2

# Generate initial map
echo "Generating initial map..."
python -c "
from src.world_model.world_model import WorldModel
from src.world_model.engine.quake3_bridge import Quake3Bridge

wm = WorldModel()
bridge = Quake3Bridge(wm)
bridge.generate_map_file('external/openarena_mod/worldmodel.map')
print('Map generated')
wm.close()
"

# Start OpenArena dedicated server
echo "Starting OpenArena server..."
openarena-server +set dedicated 2 +set net_port 27960 +map worldmodel &
SERVER_PID=$!
echo "OpenArena server started (PID: $SERVER_PID)"

# Wait a bit for server to start
sleep 3

# Start bridge sync
echo "Starting entity sync bridge..."
python -c "
from src.world_model.world_model import WorldModel
from src.world_model.engine.quake3_bridge import Quake3Bridge
import time

wm = WorldModel()
bridge = Quake3Bridge(wm)
bridge.start_sync_loop(interval=1.0)

print('Bridge running. Press Ctrl+C to stop.')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    bridge.stop_sync_loop()
    wm.close()
" &
BRIDGE_PID=$!

echo ""
echo "=== World Model 3D Viewer Running ==="
echo ""
echo "To connect:"
echo "  openarena +connect localhost:27960"
echo ""
echo "Or from another machine:"
echo "  openarena +connect $(hostname -I | awk '{print $1}'):27960"
echo ""
echo "PIDs:"
echo "  World Model API: $WM_PID"
echo "  OpenArena Server: $SERVER_PID"
echo "  Sync Bridge: $BRIDGE_PID"
echo ""
echo "To stop all:"
echo "  kill $WM_PID $SERVER_PID $BRIDGE_PID"
echo ""

# Wait for interrupt
wait
