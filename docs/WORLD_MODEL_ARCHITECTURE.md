# World Model Simulation Architecture

## Overview

A crude spatial memory system that maintains a simplified 3D representation of the AI's environment. This acts as persistent spatial memory with selective detail - similar to human mental models where we remember "desk with some items" rather than tracking all 150 objects on the desk.

## Core Concept

The world model is a **multiplayer game engine architecture** that:
- Maintains a volume of space around the AI
- Stores entities as spatial primitives (boxes, spheres, cylinders)
- Tracks relationships and importance, not exhaustive detail
- Provides spatial queries for the reasoning system
- Syncs with the AI on "core" for spatial awareness

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Core AI System                       │
│  (Different Machine - Axiom Engine, Vision, Memory)     │
└───────────────────────┬─────────────────────────────────┘
                        │ WebSocket/API
                        │ - Vision events
                        │ - Spatial queries
                        │ - Entity updates
                        ▼
┌─────────────────────────────────────────────────────────┐
│               World Model Simulation (This Machine)      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Spatial DB   │  │ Entity Store  │  │ Physics Lite │ │
│  │ (3D Grid)    │  │ (Primitives)  │  │ (Collisions) │ │
│  └──────────────┘  └───────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Importance   │  │ Query Engine  │  │ Persistence  │ │
│  │ Tracking     │  │ (Spatial)     │  │ (SQLite)     │ │
│  └──────────────┘  └───────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ Visualization │
                │ (Optional)    │
                │ - 3D View     │
                │ - Web UI      │
                └───────────────┘
```

## Entity Types

### Primitive Categories

1. **Static Environment** (low update frequency)
   - Walls, floors, ceilings
   - Furniture (desks, chairs, beds)
   - Large appliances
   - Doorways, windows

2. **Dynamic Entities** (medium update frequency)
   - Humans (position, orientation, last seen)
   - Pets
   - Mobile objects (laptop, phone, keys)
   - Robot embodiments (rover, arm)

3. **Attention Zones** (high importance)
   - Current interaction space
   - Recent activity areas
   - Remembered "important spots"

### Entity Schema

```python
{
    "id": "uuid",
    "label": "desk_main",
    "type": "furniture",
    "primitive": "box",  # box, sphere, cylinder, capsule
    "position": [x, y, z],
    "rotation": [pitch, yaw, roll],
    "scale": [width, height, depth],
    "importance": 0.85,  # 0-1, used for pruning
    "last_updated": timestamp,
    "properties": {
        "color": "brown",
        "material": "wood",
        "has_items": true,
        "item_count": "several",  # crude, not exact
        "notes": "computer desk where AI works"
    },
    "relationships": [
        {"target_id": "wall_north", "type": "adjacent_to"},
        {"target_id": "chair_main", "type": "paired_with"}
    ]
}
```

## Spatial Representation

### Coordinate System
- Origin: AI's primary location (desk/workstation)
- Units: Meters
- Y-up convention (Y is vertical)
- Range: Configurable, default ±10m cube

### Grid Structure
- Octree for efficient spatial queries
- 1m base resolution
- Adaptive subdivision for high-detail areas
- Coarse representation for distant/unimportant areas

## Importance System

### Importance Calculation
```python
importance = base_importance * recency_factor * interaction_frequency * attention_weight

where:
    base_importance = entity category weight (human=1.0, furniture=0.5, clutter=0.1)
    recency_factor = exp(-days_since_update / decay_constant)
    interaction_frequency = log(1 + interaction_count)
    attention_weight = 1.0 for current focus, decays with distance
```

### Pruning Strategy
- Keep top N% by importance (N=20% default)
- Always keep humans and safety-critical objects
- Merge similar nearby low-importance objects into zones
  - Example: "desk_surface_clutter" instead of 150 individual items
- Archive (don't delete) pruned entities for recall if mentioned

## Update Mechanisms

### From Vision System
1. reCamera detects person entering → Create/update human entity
2. Vision processor identifies object → Create/update object entity
3. Episodic memory logs event → Update spatial context

### From AI Interactions
1. AI mentions object in conversation → Boost importance
2. AI performs action → Update entity states
3. Human provides spatial information → Add/correct entities

### Autonomous Updates
1. Decay importance over time
2. Prune low-importance entities
3. Consolidate memory (daily consolidation like F7)
4. Archive old snapshots

## Query Interface

### Spatial Queries
```python
# What's near me?
world.query_radius(center=[0,0,0], radius=2.0, entity_type="all")

# What's in this direction?
world.raycast(origin=[0,0,0], direction=[1,0,0], max_distance=5.0)

# Where is X?
world.find_entity(label="partner_human")

# What's between X and Y?
world.path_entities(start_pos, end_pos)

# What can I see from here?
world.visible_from(position, fov=90)
```

### Semantic Queries
```python
# Where do I usually find X?
world.typical_location("coffee_mug")

# What room am I in?
world.current_zone(position)

# Is path clear?
world.check_clearance(start, end, radius=0.3)
```

## Integration Points

### With Core AI System

1. **Vision Events** → World Model
   - Entrance detection → Update human position
   - Object detection → Create/update entities
   - Scene changes → Trigger spatial update

2. **World Model** → Core AI Reasoning
   - Spatial context for axiom reasoning
   - "Where is partner?" queries
   - Navigation planning for rover
   - Safety checks (obstacle avoidance)

3. **Memory Systems**
   - Episodic memory includes spatial tags
   - Semantic memory references spatial patterns
   - Contact memory includes typical positions

### Communication Protocol

WebSocket API for real-time updates:
```json
{
    "type": "entity_update",
    "entity_id": "human_partner",
    "position": [2.5, 0.0, 1.0],
    "timestamp": 1702234567,
    "confidence": 0.9
}

{
    "type": "query",
    "query_type": "radius_search",
    "params": {
        "center": [0, 0, 0],
        "radius": 3.0
    }
}

{
    "type": "query_response",
    "entities": [...]
}
```

## Technology Stack

### Backend
- **Python 3.12** - Core language
- **NumPy** - Spatial calculations
- **SQLite** - Persistent storage
- **FastAPI** - WebSocket/HTTP API
- **Pydantic** - Data validation

### Optional Visualization
- **Three.js** - Web-based 3D viewer
- **WebGL** - Real-time rendering
- **Socket.IO** - Live updates to browser

### Physics (Lightweight)
- Custom collision detection (AABB, sphere)
- No full physics sim needed
- Just spatial relationships and clearance checks

## Implementation Phases

### Phase 1: Core Spatial Database
- [  ] Entity storage (SQLite schema)
- [  ] Basic primitives (box, sphere, cylinder)
- [  ] Position/rotation/scale management
- [  ] Simple importance calculation

### Phase 2: Query Engine
- [  ] Radius queries
- [  ] Raycasting
- [  ] Nearest entity search
- [  ] Zone detection

### Phase 3: Integration
- [  ] WebSocket API server
- [  ] Vision system integration
- [  ] Core AI communication
- [  ] Update handlers

### Phase 4: Intelligence
- [  ] Importance decay
- [  ] Automatic pruning
- [  ] Entity consolidation
- [  ] Pattern learning (where things usually are)

### Phase 5: Visualization (Optional)
- [  ] Web UI for 3D view
- [  ] Real-time entity updates
- [  ] Debug tools
- [  ] Manual entity editing

## Example Scenarios

### Scenario 1: Person Enters Room
```
1. reCamera detects entrance
2. Vision API sends event to World Model
3. World Model creates/updates "human_partner" entity
4. Position set to doorway coordinates
5. Importance boosted (human = 1.0)
6. Core AI can now query "where is partner?"
```

### Scenario 2: Finding Keys
```
User: "Where are my keys?"
1. Core AI queries world model: find_entity("keys")
2. World Model returns last known position + timestamp
3. If too old (> 24hr), returns "not tracked recently"
4. AI responds: "Last seen on desk_main 2 hours ago"
```

### Scenario 3: Rover Navigation
```
Rover needs to move from point A to B
1. Query: path_entities(A, B)
2. World Model returns obstacles in path
3. Rover plans around furniture
4. World Model validates clearance
```

### Scenario 4: Memory Consolidation
```
Daily at 3 AM (like F7 axiom):
1. Calculate importance for all entities
2. Prune bottom 80% that aren't safety-critical
3. Consolidate: 50 items on desk → "desk_clutter" zone
4. Archive pruned entities to cold storage
5. Log statistics to episodic memory
```

## Notes

- **Crude is good**: Don't over-model. Humans don't remember every detail.
- **Importance-driven**: Keep what matters, let the rest fade.
- **Multiplayer architecture**: Designed for multiple AI embodiments or visualization clients.
- **Persistent but forgetting**: Like human memory, things fade unless reinforced.
- **Spatial reasoning support**: Helps axiom system with physical world interactions.

## Future Extensions

- **Multi-room support**: Expand beyond single space
- **Temporal replay**: "Show me where X was yesterday"
- **Prediction**: "X is usually here at this time"
- **Shared models**: Multiple AIs share same world view
- **VR/AR integration**: Visualize AI's mental model in mixed reality
