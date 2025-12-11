#!/bin/bash
# Connect to World Model 3D viewer

echo "=== Connecting to World Model 3D Viewer ==="
echo ""
echo "Once in-game:"
echo "  1. Press ~ to open console"
echo "  2. Type: /connect localhost (should connect automatically)"
echo "  3. Use WASD to move, mouse to look"
echo "  4. Press E to interact with entities"
echo "  5. ESC for menu"
echo ""

# Check for ioquake3 first
if [ -f /home/toastee/BioMimeticAi/external/ioquake3/build/Release/ioquake3 ]; then
    echo "Launching ioquake3 client..."
    cd /home/toastee/BioMimeticAi/external/ioquake3/build/Release
    ./ioquake3 +connect localhost:27960
elif command -v openarena &> /dev/null; then
    echo "Launching OpenArena client..."
    openarena +connect localhost:27960
else
    echo "Error: No Quake 3 client found!"
    echo ""
    echo "Build ioquake3:"
    echo "  cd external/ioquake3/build && cmake .. && make"
    echo ""
    echo "Or install OpenArena:"
    echo "  sudo apt-get install openarena"
    exit 1
fi
