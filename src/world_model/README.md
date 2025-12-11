# World Model Simulation System

A crude spatial memory system that maintains a simplified 3D representation of the AI's environment using importance-based retention - similar to human mental models.

## Overview

The World Model is a **multiplayer game engine architecture** that serves as persistent spatial memory for the BioMimetic AI system running on "core". Instead of tracking every detail, it maintains only important spatial primitives with selective attention.

**Key Concept**: Like humans who remember "desk with some items" rather than all 150 objects on the desk, this system uses importance-based pruning to keep memory manageable and relevant.

## Architecture

```
Core AI (different machine)
    ↓ WebSocket/HTTP
World Model Simulation (this machine)
    ├── Spatial Database (SQLite + in-memory)
    ├── Entity Store (geometric primitives)
    ├── Importance Tracking (decay + interaction)
    └── Query Engine (radius, raycast, nearest)
```

## Features

### Entity System
- **Geometric Primitives**: Box, sphere, cylinder, capsule, plane
- **Entity Types**: Human, pet, furniture, appliance, object, zone, wall, doorway, robot
- **Importance Tracking**: 0-1 score based on recency, interaction, and type
- **Relationships**: Adjacent, on_top_of, inside, paired_with, faces, near

### Spatial Queries
- **Radius Search**: Find entities within distance
- **Nearest N**: K-nearest-neighbor search
- **Raycast**: Find entities along a ray
- **Box Query**: Entities in axis-aligned bounding box

### Memory Management
- **Importance Decay**: Entities naturally fade if not reinforced
- **Automatic Pruning**: Keep top N% or above threshold
- **Consolidation**: Daily memory cleanup (like F7 axiom)
- **Archiving**: Pruned entities saved for potential recall

### Integration
- **RESTful API**: HTTP endpoints for CRUD operations
- **WebSocket**: Real-time updates and queries
- **Vision Integration**: Updates from reCamera entrance/exit detection
- **Core AI Queries**: Spatial context for axiom reasoning

## Installation

```bash
cd /home/toastee/BioMimeticAi

# Install dependencies (if not already in main venv)
source venv/bin/activate
pip install fastapi uvicorn[standard] websockets pydantic
```

## Usage

### Start API Server

```bash
./scripts/start_world_model.sh
```

API available at: `http://localhost:8001`  
WebSocket at: `ws://localhost:8001/ws`

### Run Test Suite

```bash
source venv/bin/activate
python src/world_model/test_world_model.py
```

### Python API

```python
from src.world_model.world_model import WorldModel
from src.world_model.entity import EntityType, PrimitiveType

# Initialize
wm = WorldModel()

# Add entity
desk = wm.add_entity(
    label="desk_main",
    entity_type=EntityType.FURNITURE,
    position=(0.0, 0.8, 0.0),
    scale=(1.5, 0.1, 0.8),
    properties={"has_items": True}
)

# Query nearby
nearby = wm.query_nearby(center=(0, 0, 0), radius=5.0)

# Find entity
entity = wm.find_entity("desk_main")

# Update position
wm.update_entity(desk.id, position=(1.0, 0.8, 0.0))

# Record interaction (boosts importance)
wm.record_interaction(desk.id)

# Consolidate memory (prune low-importance)
stats = wm.consolidate_memory()

wm.close()
```

### HTTP API Examples

```bash
# Get stats
curl http://localhost:8001/stats

# Create entity
curl -X POST http://localhost:8001/entities \
  -H "Content-Type: application/json" \
  -d '{
    "label": "human_partner",
    "entity_type": "human",
    "position": {"x": 2.0, "y": 1.0, "z": 0.0},
    "primitive": "capsule",
    "scale": {"x": 0.4, "y": 1.7, "z": 0.4},
    "importance": 1.0
  }'

# Get entity by label
curl http://localhost:8001/entities/label/human_partner

# Query radius
curl -X POST http://localhost:8001/query/radius \
  -H "Content-Type: application/json" \
  -d '{
    "center": {"x": 0, "y": 0, "z": 0},
    "radius": 5.0
  }'

# Raycast
curl -X POST http://localhost:8001/query/raycast \
  -H "Content-Type: application/json" \
  -d '{
    "origin": {"x": 0, "y": 0, "z": 0},
    "direction": {"x": 1, "y": 0, "z": 0},
    "max_distance": 10.0
  }'

# Trigger consolidation
curl -X POST http://localhost:8001/consolidate
```

### WebSocket Example

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');

ws.onopen = () => {
    // Query entities within radius
    ws.send(JSON.stringify({
        type: "query_radius",
        params: {
            center: [0, 0, 0],
            radius: 5.0
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data.type, data);
};
```

## Integration with Core AI

### Vision Events → World Model

When reCamera detects entrance/exit:
```python
# In vision_api.py (on core)
import requests

response = requests.post(
    "http://world-model-host:8001/entities",
    json={
        "label": "human_partner",
        "entity_type": "human",
        "position": {"x": 2.5, "y": 1.0, "z": 1.0},
        "primitive": "capsule",
        "scale": {"x": 0.4, "y": 1.7, "z": 0.4},
        "importance": 1.0
    }
)
```

### Core AI Queries

```python
# Query spatial context for axiom reasoning
response = requests.post(
    "http://world-model-host:8001/query/radius",
    json={
        "center": {"x": 0, "y": 0, "z": 0},
        "radius": 3.0,
        "entity_type": "human"
    }
)

humans_nearby = response.json()
# Use in axiom reasoning for E1 (proximity speed)
```

## Entity Schema

```json
{
  "id": "uuid",
  "label": "desk_main",
  "entity_type": "furniture",
  "primitive": "box",
  "position": [0.0, 0.8, 0.0],
  "rotation": [0.0, 0.0, 0.0],
  "scale": [1.5, 0.1, 0.8],
  "importance": 0.75,
  "created_at": 1702234567.89,
  "last_updated": 1702238901.23,
  "interaction_count": 5,
  "properties": {
    "material": "wood",
    "has_items": true,
    "item_count": "several"
  },
  "relationships": [
    {
      "target_id": "wall_north",
      "type": "adjacent_to",
      "metadata": {}
    }
  ],
  "notes": "Main workspace desk",
  "tags": ["workspace", "important"],
  "archived": false
}
```

## Coordinate System

- **Origin**: AI's primary location (desk/workstation at 0,0,0)
- **Units**: Meters
- **Axes**: X = right, Y = up, Z = forward
- **Range**: Configurable, default ±10m cube

## Importance Calculation

```python
importance = base_importance × recency_factor × interaction_weight

where:
    base_importance = entity_type weight
        - human: 1.0
        - robot: 0.9
        - furniture: 0.5
        - object: 0.3
    
    recency_factor = exp(-days_since_update / 7.0)
    
    interaction_weight = 0.7 + 0.3 × log(1 + interaction_count)
```

## Memory Consolidation

Runs daily (or on demand) to:
1. Recalculate all importance scores
2. Prune entities below threshold (default 0.2)
3. Enforce max entity limit (default 500)
4. Archive pruned entities for potential recall

Like axiom F7 (Memory Consolidation During Idle), this transfers important spatial information while letting unimportant details fade.

## Pruning Strategy

- **Always Keep**: Humans, robots, safety-critical entities
- **Threshold**: Keep entities with importance > 0.2
- **Top N**: Keep top 20% by importance
- **Consolidation**: Merge similar low-importance entities
  - Example: 50 desk items → "desk_clutter" zone

## Future Enhancements

- [ ] Octree spatial indexing for large environments
- [ ] Multi-room support with room graphs
- [ ] Temporal replay ("where was X yesterday?")
- [ ] Predictive locations ("X is usually here at 3pm")
- [ ] Shared models between multiple AI embodiments
- [ ] 3D visualization web UI (Three.js)
- [ ] VR/AR overlay of AI's mental model
- [ ] Pattern learning (typical object locations)

## API Documentation

Full API docs available at: `http://localhost:8001/docs` (FastAPI auto-generated)

## Files

```
src/world_model/
├── __init__.py              # Package init
├── entity.py                # Entity class and enums
├── spatial_db.py            # SQLite + spatial indexing
├── world_model.py           # Main world model manager
├── api.py                   # FastAPI server
├── test_world_model.py      # Test suite
└── requirements.txt         # Dependencies

scripts/
└── start_world_model.sh     # Start API server

data/world_model/
├── spatial.db               # SQLite database
└── snapshots/               # Exported snapshots
```

## Design Philosophy

**"Crude is good"** - Don't over-model. Humans don't remember every detail, and neither should the AI. The system is designed to:

- Keep what matters (high importance)
- Let details fade naturally (exponential decay)
- Consolidate clutter (merge similar objects)
- Archive, don't delete (potential recall)
- Provide spatial reasoning, not perfect simulation

This mirrors human spatial memory: You know your desk exists, roughly where things are, and which items matter. You don't track every atom.

## See Also

- [World Model Architecture](../../docs/WORLD_MODEL_ARCHITECTURE.md) - Full design document
- [Axiom Architecture](../../AXIOM_ARCHITECTURE_README.md) - Integration with axiom reasoning
- [Vision System](../../docs/VISION_SYSTEM.md) - Integration with vision events
