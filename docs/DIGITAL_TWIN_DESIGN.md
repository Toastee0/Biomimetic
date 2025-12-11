# Digital Twin Design Specification

## Overview
The World Model serves as a **digital twin** - a persistent 3D representation of what the AI observes through its cameras. This creates spatial memory that persists beyond the current visual frame.

## Hardware Configuration
- **Camera Array**: 2 cameras on 360° rotating mast
- **Field of View**: 60° per camera
- **Depth Sensor**: 8×8 Time-of-Flight (ToF) array
- **Coverage**: Full 360° panoramic capability

## Technical Pipeline

### 1. Capture → World Model
```
[Camera + ToF] → [Vision Processing] → [Entity Detection] → [World Model Update]
```

- Cameras provide RGB imagery
- ToF provides depth information (8×8 grid)
- Vision system detects objects as **convex primitives** (boxes, spheres, cylinders, capsules)
- Detected entities update World Model spatial database

### 2. World Model → BSP Rendering
```
[Spatial Database] → [.map Generation] → [q3map2 Compilation] → [.bsp File] → [Quake 3 Engine]
```

- Spatial database contains all tracked entities
- Map generator creates `.map` file with primitives as brushes
- q3map2 compiles to `.bsp` (Binary Space Partition)
- ioquake3 engine renders the scene

**Engine Options**:
- **ioquake3**: Current implementation, standard Quake 3 with modern OS support
- **RBQUAKE-3**: Advanced alternative with modern rendering features (see below)

### 3. Digital Twin Feedback Loop
```
[Rendered Frame] → [AI Vision Input] → [Spatial Awareness] → [Action Decision]
```

- Engine renders current world model state
- Rendered frame serves as "memory" of environment
- AI can request camera movement to explore
- Persistent visualization maintains spatial context

## Key Design Principles

### 1. **Convex Primitives Only**
All objects represented as convex geometric shapes:
- **BOX**: Furniture, appliances, walls
- **SPHERE**: Balls, rounded objects
- **CYLINDER**: Cans, poles, pillars  
- **CAPSULE**: Humans, pets (pill-shaped)
- **PLANE**: Floors, ceilings, large flat surfaces

**Rationale**: Simplifies collision detection, reduces computational complexity, sufficient for spatial awareness.

### 2. **Live Insert/Remove**
Objects can be dynamically added/removed from world model:
- New detections → insert entity
- Lost track → mark for pruning (importance decay)
- Explicit removal command → delete entity
- Modified objects → update existing entity

**Workflow**:
1. Update spatial database
2. Regenerate `.map` file
3. Recompile to `.bsp`
4. Reload in engine (hot reload)

### 3. **Object Trajectory Tracking**
Each entity maintains estimated motion state:
- **Position history**: Last N positions with timestamps
- **Velocity**: Current speed vector (m/s)
- **Acceleration**: Rate of velocity change (m/s²)
- **Predicted path**: Extrapolated future positions

**Benefits**:
- Motion prediction for moving objects
- Smoother tracking of dynamic entities
- Anticipatory behavior for AI

### 4. **Structural Permanence**
Different persistence levels based on entity type:

#### **PERMANENT** (walls, floors, stairs)
- Persist indefinitely until explicitly removed
- High importance score (never pruned automatically)
- Detected once, kept forever
- Only removed by explicit "this changed" command

**Detection Strategy**:
- Floor: Large horizontal plane at lowest detected Z
- Walls: Vertical planes at consistent locations
- Stairs: Stepped vertical displacement pattern

#### **SEMI-PERMANENT** (furniture)
- Persist for extended time (hours/days)
- Moderate importance decay
- Pruned only if unseen for long period + low importance

#### **DYNAMIC** (movable objects)
- Normal importance decay
- Pruned after moderate time unseen
- Can be picked up, relocated

#### **TRANSIENT** (humans, pets)
- Fastest importance decay
- Expected to move constantly
- Short retention after leaving view

### 5. **Engine State Persistence**
The Quake 3 engine remains loaded with latest BSP:
- **Hot reload**: Update BSP without restart
- **Camera control**: AI can request pan/tilt/zoom
- **Real-time sync**: WebSocket updates for live changes
- **Render on demand**: Generate frame when needed

## Implementation Status

### ✅ Completed
- [x] Spatial database with entity storage
- [x] Geometric primitive system (BOX, SPHERE, CYLINDER, CAPSULE, PLANE)
- [x] Importance scoring and decay
- [x] Map generator (spatial DB → `.map`)
- [x] q3map2 compiler integration
- [x] BSP generation pipeline
- [x] ioquake3 engine build
- [x] Quake 3 assets installed
- [x] FastAPI + WebSocket server

### 🔄 In Progress
- [ ] Trajectory tracking per entity
- [ ] Structural permanence classification
- [ ] Vision system integration
- [ ] Camera control interface
- [ ] Hot reload mechanism

### 📋 Planned
- [ ] ToF depth sensor integration (8×8 grid)
- [ ] Rotating mast control (360° pan)
- [ ] Dual camera stereo vision
- [ ] Object convex hull detection
- [ ] Floor/wall/stair detection algorithms
- [ ] Motion prediction visualization
- [ ] Real-time BSP updates

## File Locations

```
data/world_model/
├── maps/
│   ├── worldmodel.map      # Generated map source
│   └── worldmodel.bsp      # Compiled BSP
├── test_spatial.db         # SQLite spatial database
└── checkpoints/            # Model checkpoints

src/world_model/
├── entity.py               # Entity primitives
├── spatial_db.py           # Spatial database
├── world_model.py          # Main world model
├── api.py                  # REST + WebSocket API
└── engine/
    ├── map_generator.py    # .map file generator
    └── quake3_bridge.py    # ioquake3 RCON interface

external/
├── ioquake3/               # Quake 3 engine
│   └── build/Release/
│       └── ioquake3.x86_64
└── q3map2/                 # Map compiler
    └── build/
        └── mapcompiler

scripts/
├── compile_world_map.sh    # BSP compilation script
├── start_world_3d_quake3.sh # Launch full stack
└── stop_poptartee.sh       # Shutdown
```

## API Endpoints

### REST API (Port 8001)
- `GET /entities` - List all entities
- `POST /entity` - Add/update entity
- `DELETE /entity/{id}` - Remove entity
- `GET /query/radius?x=0&y=0&z=0&radius=5` - Spatial query
- `POST /regenerate_map` - Trigger map regeneration

### WebSocket (Port 8001/ws)
- Real-time entity updates
- Camera position changes
- Importance score updates

## Usage Example

```python
from src.world_model.world_model import WorldModel
from src.world_model.engine.map_generator import MapGenerator

# Initialize world model
wm = WorldModel()

# Add detected object
from src.world_model.entity import Entity, EntityType, PrimitiveType

desk = Entity(
    label="desk_main",
    entity_type=EntityType.FURNITURE,
    primitive=PrimitiveType.BOX,
    position=(2.0, 1.0, 0.4),
    scale=(1.5, 0.8, 0.8),
    is_permanent=True  # Furniture is semi-permanent
)
wm.add_entity(desk)

# Generate and compile map
gen = MapGenerator(wm)
gen.generate('data/world_model/maps/worldmodel.map')

# Compile BSP
import subprocess
subprocess.run(['./scripts/compile_world_map.sh', 'worldmodel'])
```

## Next Steps

1. **Vision Integration**: Connect camera → vision system → entity detection
2. **Structural Detection**: Implement floor/wall/stair detection
3. **Trajectory System**: Add velocity/acceleration tracking
4. **Hot Reload**: Implement BSP hot-swapping in engine
5. **Camera Control**: API for mast rotation, camera positioning

---

## Alternative Engine: RBQUAKE-3

**Repository**: https://github.com/RobertBeckebans/RBQUAKE-3

RBQUAKE-3 is an advanced Quake 3 engine port based on XreaL, bringing Q3A up to 2009-era technology closer to Doom 3 and Quake 4 while maintaining Quake 3 gameplay. This could provide significant advantages for the digital twin visualization.

### Key Features Relevant to Digital Twin

#### Rendering Improvements
- **OpenGL 3.2 renderer**: All deprecated immediate mode calls removed
- **VBO optimization**: Vertex Buffer Objects speed up rendering of everything
- **Reduced CPU geometry processing**: Worst bottleneck with original Q3A engine eliminated
- **GPU occlusion culling**: Optional Coherent Hierarchy Culling, useful for large scenes (our spatial database could get extensive)

#### Advanced Visual Features
- **64-bit HDR lighting** with adaptive tone mapping
- **Advanced shadow mapping**: EVSM projective and omni-directional soft shadows
- **Real-time sun lights** with parallel-split shadow maps
- **Deferred shading**: Optional, better for many lights
- **Relief mapping**: Can be enabled per material

#### Model & Animation Support
- **Doom 3 .MD5mesh/.MD5anim**: Skeletal models and animations
- **Unreal Actor X .PSK/.PSA**: Additional skeletal format support
- Better for representing dynamic entities (humans, pets, robots)

#### Map Compilation
- **XBSP format**: New format with per-vertex HDR light data
- **Deluxe light mapping**: Stores dominant light direction per texel (better visual quality)
- **Two compilers included**:
  - `code/tools/xmap` - based on q3map
  - `code/tools/xmap2` - based on q3map2 (what we currently use)

#### Asset Support
- **TGA, PNG, JPG, DDS**: Multiple texture formats
- **Frame Buffer Objects (FBO)**: Offscreen rendering effects
- **TrueType fonts**: Improved support without external tools
- **Material system**: Supports Quake 3, Enemy Territory, and Doom 3 shader keywords

### Integration Considerations

**Advantages for Digital Twin**:
1. **Better performance**: VBO optimization critical for real-time updates with many entities
2. **GPU occlusion culling**: Essential when spatial database grows to city-scale
3. **Advanced lighting**: HDR with tone mapping provides better depth perception
4. **Skeletal animation**: Superior for representing humans/pets/robots with motion
5. **Deferred shading**: More efficient with many dynamic lights (one per entity based on importance)

**Implementation Path**:
- Current: `ioquake3` + `id-tech-3-tools/map-compiler` (working baseline)
- Future: `RBQUAKE-3` with `xmap2` compiler (enhanced visualization)
- Migration: Same `.map` format, compatible workflow

**Trade-offs**:
- More complex build process
- Larger codebase to maintain
- Requires OpenGL 3.2+ (modern hardware only)
- Potentially slower iteration during development

**Recommendation**: 
- Continue with ioquake3 for initial development and vision system integration
- Consider RBQUAKE-3 migration once:
  - Spatial database has 100+ entities
  - Need better performance for real-time updates
  - Want improved visual quality for AI perception
  - Require skeletal animation for dynamic entities

### Directory Structure (RBQUAKE-3)
```
RBQUAKE-3/
├── base/                  # Media directory (models, textures, sounds, maps)
├── code/                  # Source code
│   ├── renderer/         # OpenGL 3.2 renderer
│   ├── game/             # Game logic
│   ├── tools/
│   │   ├── xmap/         # Map compiler (q3map-based)
│   │   ├── xmap2/        # Map compiler (q3map2-based) ← We'd use this
│   │   └── xmaster/      # Master server
├── blender/              # Blender plugins (ase, md3, md5 models)
```

### Potential Enhancements with RBQUAKE-3

1. **Entity Importance Lighting**: Each entity gets dynamic light scaled by importance score (HDR makes this visually meaningful)
2. **Motion Trails**: Use relief mapping to show predicted trajectories
3. **Skeletal Entities**: Represent humans/pets with proper animated skeletons instead of capsules
4. **Depth Perception**: Better shadows and lighting improve AI's spatial understanding
5. **Large-Scale Scenes**: GPU occlusion culling enables city-block scale spatial memory
