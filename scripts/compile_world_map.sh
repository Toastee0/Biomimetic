#!/bin/bash
# Compile World Model map to BSP for Quake 3 visualization
#
# Usage: ./scripts/compile_world_map.sh [map_name]
#
# This compiles a .map file through all stages:
# 1. BSP compilation (geometry processing)
# 2. VIS processing (visibility calculation)
# 3. LIGHT processing (lightmap generation)

set -e

MAP_NAME=${1:-worldmodel}
MAP_FILE="data/world_model/maps/${MAP_NAME}.map"
OUTPUT_DIR="data/world_model/maps"
Q3MAP2="external/q3map2/build/mapcompiler"
FS_BASEPATH="$HOME/.q3a"

# Check if map file exists
if [ ! -f "$MAP_FILE" ]; then
    echo "Error: Map file not found: $MAP_FILE"
    exit 1
fi

# Check if q3map2 is built
if [ ! -f "$Q3MAP2" ]; then
    echo "Error: q3map2 not found. Build it first:"
    echo "  cd external/q3map2 && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=RELEASE .. && make -j\$(nproc)"
    exit 1
fi

echo "=== Compiling ${MAP_NAME}.map to BSP ==="
echo ""

# Stage 1: BSP compilation
echo "Stage 1/3: BSP compilation (geometry processing)"
$Q3MAP2 -bsp \
    -fs_basepath "$FS_BASEPATH" \
    -game quake3 \
    -meta \
    -v \
    "$MAP_FILE"

echo ""

# Stage 2: VIS processing (visibility/portalization)
echo "Stage 2/3: VIS processing (visibility calculation)"
$Q3MAP2 -vis \
    -fs_basepath "$FS_BASEPATH" \
    -game quake3 \
    -v \
    "${OUTPUT_DIR}/${MAP_NAME}.bsp"

echo ""

# Stage 3: LIGHT processing (lightmap generation)
echo "Stage 3/3: LIGHT processing (lightmap generation)"
$Q3MAP2 -light \
    -fs_basepath "$FS_BASEPATH" \
    -game quake3 \
    -fast \
    -filter \
    -v \
    "${OUTPUT_DIR}/${MAP_NAME}.bsp"

echo ""
echo "=== Compilation complete ==="
echo "Output: ${OUTPUT_DIR}/${MAP_NAME}.bsp"
echo ""
echo "To test in ioquake3:"
echo "  cp ${OUTPUT_DIR}/${MAP_NAME}.bsp ~/.q3a/baseq3/maps/"
echo "  cd external/ioquake3/build/Release"
echo "  ./ioquake3.x86_64 +set sv_pure 0 +map ${MAP_NAME}"
