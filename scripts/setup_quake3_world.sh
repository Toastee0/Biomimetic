#!/bin/bash
# Setup script for Quake 3 World Model integration

set -e

cd /home/toastee/BioMimeticAi

echo "=== Setting up Quake 3 World Model Viewer ==="
echo ""

# Check if external directory exists
mkdir -p external

# Option 1: Use OpenArena (fully free, no Quake 3 assets needed)
echo "Option 1: OpenArena (Recommended - Fully Free)"
echo "  - No Quake 3 purchase needed"
echo "  - All assets open source"
echo "  - Same engine as Quake 3"
echo ""
echo "Install command:"
echo "  sudo apt-get install openarena openarena-server"
echo ""

# Option 2: ioquake3 + Quake 3 assets
echo "Option 2: ioquake3 + Quake 3 Assets"
echo "  - Requires Quake 3 Arena game files"
echo "  - Better original assets"
echo ""
echo "Install ioquake3:"
if [ ! -d "external/ioquake3" ]; then
    echo "  Cloning ioquake3..."
    git clone https://github.com/ioquake/ioq3.git external/ioquake3
    cd external/ioquake3
    echo "  Building..."
    make
    cd ../..
    echo "  ✓ ioquake3 built"
else
    echo "  ✓ ioquake3 already installed"
fi
echo ""
echo "You'll need to copy pak0.pk3 from Quake 3 to:"
echo "  ~/.q3a/baseq3/pak0.pk3"
echo ""

# Option 3: Godot Engine
echo "Option 3: Godot Engine (Easiest to customize)"
echo "  Download from: https://godotengine.org/"
echo "  Or: sudo snap install godot"
echo ""

echo "=== Which option do you want? ==="
echo "1) OpenArena (recommended, easiest)"
echo "2) ioquake3 (classic, needs Q3 assets)"
echo "3) Godot (modern, most customizable)"
echo "4) Skip for now"
echo ""
read -p "Choice (1-4): " choice

case $choice in
    1)
        echo "Installing OpenArena..."
        sudo apt-get update
        sudo apt-get install -y openarena openarena-server
        echo "✓ OpenArena installed"
        echo ""
        echo "To start:"
        echo "  ./scripts/start_world_3d_openarena.sh"
        ;;
    2)
        echo "ioquake3 is built. Copy your Quake 3 pak files to ~/.q3a/baseq3/"
        echo ""
        echo "To start:"
        echo "  ./scripts/start_world_3d_quake3.sh"
        ;;
    3)
        echo "Install Godot:"
        echo "  sudo snap install godot"
        echo "  or download from https://godotengine.org/"
        echo ""
        echo "Then run:"
        echo "  ./scripts/setup_godot_world.sh"
        ;;
    4)
        echo "Skipping 3D engine setup"
        ;;
esac

echo ""
echo "=== Setup Complete ==="
