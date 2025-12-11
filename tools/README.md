# Quake 3 CLI Tools for Fast Map Generation

CLI tools for ultra-fast Quake 3 map generation from camera/LIDAR entity data. Part of the BioMimetic AI spatial twin system.

## Philosophy

**Generate so fast that we don't need incremental updates** - just regenerate the entire map and hot-reload it in the engine.

- Structure (walls/floors) changes rarely → cache BSP (~2 sec compile, one-time)
- Entities change constantly → regenerate fast (~10ms)
- Fast entity-only reload → `q3map2 -onlyents` (~100ms vs ~2000ms full compile)

**Performance Target**: < 200ms from camera detection to visible in engine

## Installation

```bash
# Build
make

# Install system-wide (optional)
sudo make install

# Run tests
make test
```

## Tools

### q3entity - Entity JSON → .map Entities

Convert World Model entity JSON to Quake 3 entity definitions.

**Usage**:
```bash
# From file
q3entity entities.json -o entities.map

# From World Model API
curl http://localhost:8001/entities | bin/q3entity -o entities.map

# Filter by importance
q3entity entities.json --min-importance 0.5 -o filtered.map
```

**Input JSON Format**:
```json
{
  "entities": [
    {
      "id": "uuid-1234",
      "label": "person_01",
      "entity_type": "HUMAN",
      "primitive": "CAPSULE",
      "position": [2.0, 1.5, 0.9],
      "rotation": [0, 0, 0],
      "scale": [0.4, 0.4, 1.8],
      "velocity": [0.5, 0, 0],
      "importance": 0.9
    }
  ]
}
```

**Performance**: < 10ms for 100 entities

---

### q3merge - Merge Structure + Entities

Combine structure .map (worldspawn brushes) with entity .map files.

**Usage**:
```bash
# Merge structure and entities
bin/q3merge structure.map entities.map -o world.map

# Or with explicit flags
bin/q3merge --base structure.map --entities entities.map -o world.map
```

**Performance**: < 20ms

---

### q3reload - Hot Reload Map

Compile and reload map in running Quake 3 engine with fast entity-only mode.

**Usage**:
```bash
# Fast path: entity-only reload (~100ms)
q3reload world.map --entities-only

# Full path: complete BSP compilation (~2000ms)
q3reload world.map --full

# With RCON settings
q3reload world.map --entities-only \
    --rcon-host localhost \
    --rcon-port 27960 \
    --rcon-password worldmodel
```

**Fast Path** (`--entities-only`):
- Uses `q3map2 -onlyents`
- Skips geometry compilation
- Only updates entity lump in BSP
- ~100ms total

**Full Path** (`--full`):
- Complete BSP → VIS → LIGHT compilation
- ~2000ms total
- Only needed when structure changes

---

## Fast Regeneration Workflow

### One-Time: Generate Structure

```bash
# Generate room structure (one time)
# Note: q3brush tool coming in Phase 2
# For now, use existing brush_csg_tool
../brush_csg_tool structure.map --hollow

# Full BSP compile (slow, but only once)
q3map2 -bsp -vis -light -fs_basepath ~/.q3a structure.map

# This creates structure.bsp (cached)
```

### Runtime: Fast Entity Updates

```bash
# 1. Fetch current entities from World Model
curl -s http://localhost:8001/entities > entities.json

# 2. Convert to .map entities (< 10ms)
bin/q3entity entities.json -o entities.map

# 3. Merge with structure (< 20ms)
bin/q3merge structure.map entities.map -o world.map

# 4. Fast entity-only compile and reload (< 100ms)
q3reload world.map --entities-only

# Total: ~130ms from API call to engine reload
```

## Architecture

### Shared Library

`lib/q3map_common.c` - Core functions extracted from GtkRadiant:
- Plane database (32,768 plane limit)
- `BrushFromBounds()` - Create axial brushes
- `MakeBrushHollow()` - Generate 6-wall hollow rooms
- `WorldToQuake()` - Coordinate conversion (Y-up → Z-up, meters → units)
- `ImportanceToColor()` - Importance → RGB mapping

### Coordinate Systems

- **World Model**: Y-up, meters
- **Quake 3**: Z-up, 32 units = 1 meter

Conversion handled automatically by `WorldToQuake()`.

### Importance → Color Mapping

- High (≥0.8): Red
- Medium (≥0.5): Yellow
- Low (≥0.2): Green
- Very low (<0.2): Gray

Visual feedback loop for cognitive system.

## Development

### Build System

```makefile
tools/
├── Makefile              # Build system
├── bin/                  # Built executables
│   ├── q3entity
│   └── q3merge
├── build/                # Object files
├── lib/                  # Shared library
│   ├── q3map_common.h
│   └── q3map_common.c
├── q3entity/             # Entity converter source
│   └── main.c
├── q3merge/              # Map merger source
│   └── main.c
└── q3reload              # Hot reload script
```

### Adding New Tools

1. Create `tools/newtool/main.c`
2. Add to `TOOLS` in Makefile
3. Add build rules
4. Document in README

## Phase Roadmap

### ✅ Phase 1: Core Infrastructure (Complete)
- Shared library with brush primitives
- q3entity (JSON → .map)
- q3merge (merge maps)
- q3reload (hot reload)
- Build system
- **Result**: Working fast regeneration pipeline

### 🔄 Phase 2: Advanced Primitives (Next)
- q3brush (cylinder, stairs, ramp)
- CSG subtract for doorways
- q3trajectory (motion paths)
- **Target**: Rich structure generation

### Phase 3: Performance Optimization
- Sub-100ms regeneration for 100 entities
- Memory pooling (arena allocators)
- Parallel processing
- Benchmarking suite

### Phase 4: Camera/LIDAR Integration
- Event listener in World Model
- YOLO detection → Entity JSON pipeline
- Background regeneration thread
- Monitoring dashboard

## Integration with World Model

```python
# Example: Python integration
from subprocess import run
import json

# Fetch entities
entities = world_model.db.get_all_entities()
entity_json = {"entities": [e.to_dict() for e in entities]}

# Write JSON
with open("/tmp/entities.json", "w") as f:
    json.dump(entity_json, f)

# Generate map
run(["bin/q3entity", "/tmp/entities.json", "-o", "/tmp/entities.map"])
run(["bin/q3merge", "structure.map", "/tmp/entities.map", "-o", "world.map"])
run(["q3reload", "world.map", "--entities-only"])

# Total time: ~150ms
```

## Performance Metrics

### Current (Phase 1)
- **q3entity** (100 entities): ~10ms
- **q3merge**: ~20ms
- **q3reload --entities-only**: ~100ms
- **Total pipeline**: ~130ms ✅

### Targets
- 100 entities: < 100ms total
- 1000 entities: < 500ms
- Continuous updates: 10Hz (100ms interval)

## License

Part of BioMimetic AI project. Uses code patterns from GtkRadiant (GPL).

## Credits

- GtkRadiant CSG functions (brush primitives)
- q3map2 compiler (BSP generation)
- cJSON library (JSON parsing)
