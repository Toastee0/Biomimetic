# 3D Engine Integration for World Model

## Overview

The World Model spatial database is rendered as a **playable 3D world** using an open-source game engine. This allows you to:
- **Observe** the AI's spatial mental model in real-time 3D
- **Navigate** through the AI's perception of space
- **Edit** entities directly in the 3D environment
- **Multiplayer** support for multiple observers/editors

## Engine Options

### Option 1: ioquake3 (Recommended)
- **Engine**: Quake III Arena (id Tech 3)
- **Status**: Actively maintained, modern fork
- **Pros**: Excellent multiplayer, well-documented, fast
- **Cons**: Requires learning Quake 3 modding
- **License**: GPLv2

### Option 2: DarkPlaces
- **Engine**: Enhanced Quake engine
- **Status**: Active, modern graphics
- **Pros**: Beautiful rendering, simpler than Q3
- **Cons**: Less robust multiplayer
- **License**: GPLv2

### Option 3: Godot Engine
- **Engine**: Modern game engine
- **Status**: Very active, user-friendly
- **Pros**: Easy to work with, Python-like scripting
- **Cons**: More setup needed, overkill for this
- **License**: MIT

## Architecture with ioquake3

```
┌─────────────────────────────────────────────────────┐
│              World Model System (Python)            │
│  - Entity Storage (SQLite)                          │
│  - Spatial Queries                                  │
│  - Importance Tracking                              │
└────────────────────┬────────────────────────────────┘
                     │
                     │ Bridge Module (Python → Quake)
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           ioquake3 Game Server (C)                  │
│  - BSP World Renderer                               │
│  - Entity Spawning                                  │
│  - Multiplayer Server                               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│        Client(s) - Quake 3 Renderer                 │
│  - Walk through AI's mental model                   │
│  - See entities as 3D objects                       │
│  - Select/edit entities                             │
│  - Multiple observers simultaneously                │
└─────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Bridge Module
Create Python bridge that:
1. Reads entities from World Model DB
2. Generates Quake 3 map format (.map or .bsp)
3. Spawns dynamic entities via game server protocol
4. Listens for client commands (entity selection/editing)

### Phase 2: Entity Mapping
Map World Model primitives to Quake 3 entities:
- **Box** → `func_door` or custom brush
- **Sphere** → `misc_model` (sphere.md3)
- **Cylinder** → `misc_model` (cylinder.md3)
- **Capsule** → `misc_model` (capsule.md3)
- **Human** → `player` model
- **Robot** → Custom bot model

### Phase 3: Real-time Sync
- World Model updates → Server entity updates
- Client selections → World Model queries
- Client edits → World Model database writes

### Phase 4: Multiplayer Features
- Multiple observers
- Shared editing
- Chat integration with Discord bot
- VR support (later)

## Files Structure

```
src/world_model/
├── engine/
│   ├── __init__.py
│   ├── quake3_bridge.py      # Python ↔ Quake bridge
│   ├── map_generator.py      # Generate .map files
│   ├── entity_spawner.py     # Dynamic entity spawning
│   └── server_controller.py  # Control ioquake3 server
├── models/                    # 3D models for entities
│   ├── human.md3
│   ├── robot.md3
│   ├── furniture/
│   └── primitives/
└── maps/
    ├── worldmodel.map         # Generated map file
    └── worldmodel.bsp         # Compiled BSP

external/
├── ioquake3/                  # Git submodule
│   ├── build/
│   └── ...
└── q3asm/                     # BSP compiler tools
```

## Installation

### 1. Install ioquake3

```bash
cd /home/toastee/BioMimeticAi

# Clone ioquake3
git submodule add https://github.com/ioquake/ioq3.git external/ioquake3
cd external/ioquake3

# Build
make

# Copy base game data (you'll need Quake 3 pak files or use OpenArena)
mkdir -p ~/.q3a/baseq3
# Copy pak0.pk3 from Quake 3 or use OpenArena free assets
```

### 2. Install OpenArena (Free Alternative)

```bash
sudo apt-get install openarena
# Uses same engine, fully open-source assets
```

### 3. Install Map Compiler Tools

```bash
sudo apt-get install q3map2
```

## Usage

### Start 3D World View

```bash
# Terminal 1: Start World Model API
./scripts/start_world_model.sh

# Terminal 2: Start Quake 3 Server Bridge
./scripts/start_world_3d.sh

# Terminal 3: Connect with Quake 3 Client
./scripts/connect_world_view.sh
```

### In-Game Controls

- **Walk around**: WASD keys
- **Look**: Mouse
- **Select entity**: Left-click
- **Edit entity**: Press E (opens edit dialog)
- **Toggle entity labels**: Tab
- **Toggle importance visualization**: I (color-code by importance)
- **Console**: ~ (for queries)

### Console Commands

```
/wm_query_radius 5.0          # Show entities within 5m
/wm_find human_partner        # Highlight entity
/wm_spawn furniture desk 0 0 0  # Add entity at position
/wm_delete entity_id          # Remove entity
/wm_set_importance entity_id 0.8  # Adjust importance
/wm_consolidate               # Trigger memory pruning
```

## Entity Visualization

### Importance Coloring
- **Red**: High importance (1.0 - 0.8)
- **Yellow**: Medium (0.79 - 0.5)
- **Green**: Low (0.49 - 0.2)
- **Gray**: Very low (< 0.2, will be pruned)

### Entity Labels
Floating text above entities showing:
- Label
- Type
- Importance score
- Distance from viewer

### Interaction Indicators
- **Glowing**: Recently interacted with
- **Pulsing**: Currently being queried by AI
- **Fading**: Low importance, will be pruned soon

## Map Generation

The bridge automatically generates a Quake 3 map from World Model data:

```python
# Example generated .map file
{
"classname" "worldspawn"
"message" "AI World Model"
// Floor
{
brushDef
{
( 0 0 1 0 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) common/caulk 0 0 0
...
}
}
}
// Entities
{
"classname" "misc_model"
"model" "models/furniture/desk.md3"
"origin" "0 0 24"
"angle" "0"
"_color" "0.7 0.5 0.3"  // Importance-based color
}
```

## Protocol Specification

### World Model → Quake Server

```json
{
  "type": "spawn_entity",
  "entity_id": "uuid",
  "label": "desk_main",
  "model": "models/furniture/desk.md3",
  "position": [0, 0, 24],
  "rotation": [0, 0, 0],
  "scale": [1.5, 0.8, 0.1],
  "color": [0.7, 0.5, 0.3],
  "properties": {}
}
```

### Quake Client → World Model

```json
{
  "type": "entity_selected",
  "entity_id": "uuid",
  "player_id": "player1"
}

{
  "type": "entity_moved",
  "entity_id": "uuid",
  "new_position": [1.0, 0.5, 24],
  "player_id": "player1"
}
```

## Alternative: Simpler Approach with Godot

If ioquake3 is too complex, we could use **Godot Engine** which has:
- Built-in Python-like scripting (GDScript)
- Easy 3D scene management
- WebSocket support out of the box
- Simpler to integrate

```gdscript
# Godot GDScript example
extends Node3D

var websocket = WebSocketClient.new()

func _ready():
    websocket.connect_to_url("ws://localhost:8001/ws")
    websocket.message_received.connect(_on_message)

func _on_message(message):
    var data = JSON.parse(message)
    if data.type == "entity_created":
        spawn_entity(data.entity)

func spawn_entity(entity):
    var mesh = MeshInstance3D.new()
    # Create mesh based on primitive type
    # Position and scale based on entity data
    add_child(mesh)
```

## Quick Start: Choose Your Engine

### For ioquake3 (True multiplayer, classic FPS feel)
```bash
./scripts/setup_quake3_world.sh
```

### For Godot (Easier to customize, modern)
```bash
./scripts/setup_godot_world.sh
```

### For web-based Three.js (No install needed)
```bash
# Just open http://localhost:8001/viewer
# Already implemented in API server
```

## Next Steps

Which would you prefer?
1. **ioquake3** - Classic Quake 3 engine, true multiplayer FPS
2. **OpenArena** - Same as ioquake3 but with free assets
3. **Godot** - Modern engine, easier to customize
4. **Three.js web viewer** - Simplest, no game engine install

Let me know and I'll implement the full integration!
