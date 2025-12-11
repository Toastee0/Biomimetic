#!/bin/bash
# Launch Quake 3 in Windows from WSL
# This avoids the Linux mouse issues

cd /mnt/m/wslprojects/ioquake3
./ioquake3.x86_64.exe +set sv_pure 0 +set com_hunkMegs 256 +map q3dm1 &

echo "ioquake3 launched in Windows!"
echo ""
echo "Controls:"
echo "  WASD - Move"
echo "  Mouse - Look around"
echo "  Space - Jump"
echo "  ~ - Console"
echo "  ESC - Menu"
echo ""
echo "Console commands:"
echo "  /map q3dm1      - Load default map"
echo "  /map worldmodel - Load our custom map (once we fix it)"
echo "  /noclip         - Fly mode"
echo "  /god            - God mode"
echo "  /quit           - Exit"
