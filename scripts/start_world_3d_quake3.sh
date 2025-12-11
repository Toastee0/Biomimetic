#!/bin/bash
# Start World Model with ioquake3 dedicated server and bridge

cd /home/toastee/BioMimeticAi

echo "=== Starting World Model 3D Viewer (ioquake3) ==="
echo ""

# Check if ioquake3 is built
if [ ! -f "external/ioquake3/build/Release/ioq3ded" ]; then
    echo "Error: ioquake3 not built!"
    echo "Run: cd external/ioquake3/build && cmake .. && make"
    exit 1
fi

# Check if Quake 3 data files exist
if [ ! -f "$HOME/.q3a/baseq3/pak0.pk3" ]; then
    echo "Error: Quake 3 data files not found!"
    echo "Copy pak files to ~/.q3a/baseq3/"
    exit 1
fi

# Kill any existing processes
pkill -f "ioquake3.*worldmodel" || true
pkill -f "uvicorn.*world_model" || true

# Start World Model API in background
echo "Starting World Model API server..."
source venv/bin/activate
python -m uvicorn src.world_model.api:app --host 0.0.0.0 --port 8001 > logs/world_model_api.log 2>&1 &
WM_PID=$!
echo "  ✓ World Model API started (PID: $WM_PID)"

# Wait for API to start
sleep 3

# Create worldmodel directory in baseq3
mkdir -p ~/.q3a/worldmodel

# Generate initial test entities
echo "Creating test world..."
python << 'EOF'
import sys
sys.path.insert(0, '/home/toastee/BioMimeticAi/src')

from world_model.world_model import WorldModel
from world_model.entity import EntityType, PrimitiveType

# Create world model with test data
wm = WorldModel(db_path="data/world_model/spatial.db")

# Clear any existing entities for fresh start
# (you can comment this out to keep existing entities)
# for entity in wm.db.get_all_entities():
#     wm.db.delete_entity(entity.id)

# Add some test entities if world is empty
if len(wm.db.get_all_entities()) == 0:
    print("  Creating test environment...")
    
    # Add floor plane
    wm.add_entity(
        label="floor",
        entity_type=EntityType.ZONE,
        position=(0.0, 0.0, 0.0),
        primitive=PrimitiveType.PLANE,
        scale=(20.0, 20.0, 0.1),
        importance=0.8
    )
    
    # Add walls
    wm.add_entity(
        label="wall_north",
        entity_type=EntityType.WALL,
        position=(0.0, -5.0, 1.5),
        primitive=PrimitiveType.BOX,
        scale=(10.0, 0.2, 3.0),
        importance=0.6
    )
    
    # Add desk
    wm.add_entity(
        label="desk_main",
        entity_type=EntityType.FURNITURE,
        position=(0.0, 0.0, 0.8),
        primitive=PrimitiveType.BOX,
        scale=(1.5, 0.8, 0.1),
        properties={"type": "desk"},
        importance=0.7
    )
    
    # Add human
    wm.add_entity(
        label="human_partner",
        entity_type=EntityType.HUMAN,
        position=(2.0, 0.0, 1.0),
        primitive=PrimitiveType.CAPSULE,
        scale=(0.4, 0.4, 1.7),
        importance=1.0
    )
    
    # Add robot
    wm.add_entity(
        label="rover_bot",
        entity_type=EntityType.ROBOT,
        position=(-1.0, 0.5, 0.2),
        primitive=PrimitiveType.BOX,
        scale=(0.3, 0.4, 0.2),
        importance=0.9
    )
    
    print(f"  ✓ Created {len(wm.db.get_all_entities())} test entities")
else:
    print(f"  ✓ World has {len(wm.db.get_all_entities())} existing entities")

wm.close()
EOF

echo ""
echo "Generating World Model map file..."
python << 'MAPGEN'
import sys
sys.path.insert(0, '/home/toastee/BioMimeticAi/src')

from world_model.world_model import WorldModel
from world_model.engine.map_generator import MapGenerator

wm = WorldModel(db_path="data/world_model/spatial.db")
generator = MapGenerator(wm)
generator.generate("data/world_model/maps/worldmodel.map")
wm.close()
print("  ✓ Map file generated")
MAPGEN

echo ""
echo "Starting ioquake3 dedicated server..."
cd external/ioquake3/build/Release

# Start dedicated server with standard map (we'll load custom map later)
./ioq3ded \
    +set dedicated 2 \
    +set net_port 27960 \
    +set sv_hostname "BioMimetic World Model" \
    +set g_gametype 0 \
    +set bot_enable 0 \
    +set sv_pure 0 \
    +map q3dm1 \
    > /home/toastee/BioMimeticAi/logs/ioquake3_server.log 2>&1 &
SERVER_PID=$!

cd /home/toastee/BioMimeticAi
echo "  ✓ ioquake3 server started (PID: $SERVER_PID)"

# Wait for server to start
sleep 5

echo ""
echo "=== World Model 3D Viewer Running ==="
echo ""
echo "Server Info:"
echo "  World Model API: http://localhost:8001"
echo "  Quake 3 Server:  localhost:27960"
echo ""
echo "AI Camera:"
echo "  Origin: ${world_model.origin:-0,0,0}"
echo "  The glowing blue entity marks the AI's viewpoint"
echo ""
echo "Controls:"
echo "  F1 - Free spectator mode (fly around)"
echo "  F2 - Follow AI camera"
echo "  WASD - Move"
echo "  Mouse - Look"
echo ""
echo "To connect:"
echo "  ./scripts/connect_world_view.sh"
echo ""
echo "Or from another machine:"
echo "  ioquake3 +connect $(hostname -I | awk '{print $1}'):27960"
echo ""
echo "Processes:"
echo "  API PID:    $WM_PID"
echo "  Server PID: $SERVER_PID"
echo ""
echo "Logs:"
echo "  tail -f logs/world_model_api.log"
echo "  tail -f logs/ioquake3_server.log"
echo ""
echo "To stop:"
echo "  pkill -f ioquake3"
echo "  pkill -f \"uvicorn.*world_model\""
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for interrupt
trap "echo 'Stopping...'; kill $WM_PID $SERVER_PID 2>/dev/null; exit" INT TERM
wait
