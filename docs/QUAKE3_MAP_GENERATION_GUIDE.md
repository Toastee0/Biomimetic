# Quake 3 Map Generation Guide

## Understanding Brush Geometry

### Core Concepts

**Brush**: A convex volume defined by the **intersection** of half-spaces. Think of it like carving a shape by cutting away with infinite planes.

**Plane**: Defined by 3 non-collinear points that lie on the plane surface. The plane equation divides 3D space into two half-spaces.

**Convexity Requirement**: Every brush MUST be convex. Non-convex shapes require multiple brushes.

### Plane Definition Format

```
( x1 y1 z1 ) ( x2 y2 z2 ) ( x3 y3 z3 ) texture offsetX offsetY rotation scaleX scaleY contentFlags surfaceFlags value
```

**Components:**
- **3 Points**: Must lie on the plane (not necessarily brush corners)
- **Texture**: Path to shader/texture (e.g., `base_floor/concrete`)
- **UV Mapping**: `offsetX`, `offsetY`, `rotation`, `scaleX`, `scaleY`
- **Flags**: `contentFlags` (0), `surfaceFlags` (0), `value` (0) - usually 0

**Example:**
```
( 424 48 -304 ) ( 424 -80 -304 ) ( 424 48 312 ) base_floor/clang_floor2 0 0 0 1 1 0 0 0
```
All three points have x=424, so this defines the plane x=424 (perpendicular to X-axis).

### Brush Construction Pattern

A sealed box requires **6 brushes** (one slab for each face):

```
Brush 1: Floor (bottom slab)
  - 5 planes shared with neighboring brushes
  - 1 unique plane: bottom face at z=min_z
  - Thickness: typically 8-16 units

Brush 2: Ceiling (top slab)
  - 5 shared planes
  - 1 unique plane: top face at z=max_z
  - Thickness: 8-16 units

Brushes 3-6: Walls (side slabs)
  - Similar pattern
  - Each has 1 unique plane defining that wall's inner surface
```

### How Planes Share Between Brushes

When two brushes need to align perfectly:

**Floor brush and left wall must share the edge at x=min_x:**
- Floor has plane: `( min_x min_y min_z ) ( min_x max_y min_z ) ( min_x min_y max_z )`
- Left wall has same plane with different ordering/orientation

The **intersection** of all planes in each brush defines the brush volume. Shared planes ensure no gaps.

### Example: Simple Sealed Room

```
// Room: 512x512x288 units
// Floor thickness: 16 units (z=-16 to z=0)
// Ceiling thickness: 16 units (z=256 to z=272)
// Wall thickness: 8 units

{
"classname" "worldspawn"

// Brush 0: Floor
{
( -256 -256 -16 ) ( 256 -256 -16 ) ( -256 256 -16 ) base_floor/concrete 0 0 0 1 1 0 0 0
( -256 256 0 ) ( 256 256 0 ) ( -256 -256 0 ) base_floor/concrete 0 0 0 1 1 0 0 0
( -256 -256 -16 ) ( -256 256 -16 ) ( -256 -256 0 ) common/caulk 0 0 0 1 1 0 0 0
( 256 256 -16 ) ( 256 -256 -16 ) ( 256 256 0 ) common/caulk 0 0 0 1 1 0 0 0
( 256 -256 -16 ) ( -256 -256 -16 ) ( 256 -256 0 ) common/caulk 0 0 0 1 1 0 0 0
( -256 256 -16 ) ( 256 256 -16 ) ( -256 256 0 ) common/caulk 0 0 0 1 1 0 0 0
}

// Brush 1: Ceiling
{
( -256 256 256 ) ( 256 256 256 ) ( -256 -256 256 ) base_floor/concrete 0 0 0 1 1 0 0 0
( -256 -256 272 ) ( 256 -256 272 ) ( -256 256 272 ) base_floor/concrete 0 0 0 1 1 0 0 0
( -256 -256 256 ) ( -256 256 256 ) ( -256 -256 272 ) common/caulk 0 0 0 1 1 0 0 0
( 256 256 256 ) ( 256 -256 256 ) ( 256 256 272 ) common/caulk 0 0 0 1 1 0 0 0
( 256 -256 256 ) ( -256 -256 256 ) ( 256 -256 272 ) common/caulk 0 0 0 1 1 0 0 0
( -256 256 256 ) ( 256 256 256 ) ( -256 256 272 ) common/caulk 0 0 0 1 1 0 0 0
}

// Brush 2: Back wall (y = -256)
{
( -256 -256 0 ) ( 256 -256 0 ) ( -256 -256 256 ) base_wall/concrete 0 0 0 1 1 0 0 0
( 256 -264 0 ) ( -256 -264 0 ) ( 256 -264 256 ) common/caulk 0 0 0 1 1 0 0 0
( -256 -256 0 ) ( -256 -256 256 ) ( -256 -264 0 ) common/caulk 0 0 0 1 1 0 0 0
( 256 -256 256 ) ( 256 -256 0 ) ( 256 -264 256 ) common/caulk 0 0 0 1 1 0 0 0
( 256 -256 0 ) ( -256 -256 0 ) ( 256 -264 0 ) common/caulk 0 0 0 1 1 0 0 0
( -256 -256 256 ) ( 256 -256 256 ) ( -256 -264 256 ) common/caulk 0 0 0 1 1 0 0 0
}

// ... similar for other 3 walls
}
```

## Common Mistakes and Solutions

### Problem: 0 Surfaces / Invalid Geometry

**Cause**: Planes don't form a valid convex volume
- Non-coplanar points used to define plane
- Planes intersect incorrectly
- Brush is concave (impossible!)

**Solution**:
1. Use simple axis-aligned points for learning
2. Each plane should define ONE flat surface
3. Verify all 3 points lie on intended plane
4. Check that intersection of all planes creates enclosed volume

### Problem: Leaks / Non-Sealed Map

**Cause**: Gaps between brushes
- Brushes don't share edges properly
- Floating point precision issues

**Solution**:
- Use integer coordinates (multiples of 8 or 16)
- Ensure neighboring brushes share exact same plane definitions
- Test with simple box first before complex shapes

### Problem: 99999 Bounds / Infinite Brush

**Cause**: Planes don't intersect to form finite volume
- Missing planes
- Parallel planes facing same direction

**Solution**:
- Box needs exactly 6 planes (one pair per axis)
- Check plane normals point inward

## Texture Best Practices

### Visible vs Hidden Faces

**Visible faces**: Use actual textures
```
base_floor/concrete    (floor)
base_wall/concrete     (walls)
base_floor/clang_floor2 (metal)
```

**Hidden faces**: Use `common/caulk` (optimization - not rendered)
```
common/caulk           (backs of walls, interior faces)
```

### Texture Naming Convention

Format: `category/name`
- `base_floor/`: Floor textures
- `base_wall/`: Wall textures
- `common/`: Special shaders (caulk, clip, nodraw, etc.)
- `textures/`: Custom textures

## Creating Connected Rooms

To connect two rooms with a hallway:

1. **Room 1**: Standard sealed box (6 brushes)
2. **Hallway**: Box with OPEN ends (4 brushes - floor, ceiling, 2 walls)
3. **Room 2**: Standard sealed box (6 brushes)

**Key**: Hallway brushes share planes with room brushes at connection points.

```
Room 1 right wall at x=0 → omit this wall, hallway starts at x=0
Hallway left boundary: x=0 (open - no brush)
Hallway right boundary: x=384 (open - no brush)
Room 2 left wall at x=384 → omit this wall, hallway ends at x=384
```

Result: 5 + 4 + 5 = **14 brushes** total

## Scale and Units

- **Quake 3**: Z-up, 32 units = 1 meter
- **Standard grid**: 8 or 16 unit increments
- **Typical dimensions**:
  - Floor height: 0 (z=0)
  - Ceiling height: 256 (8 feet = 2.44m)
  - Doorway: 128 units wide × 256 high
  - Hallway: 256-384 units wide
  - Room: 512-1024 units per side

## Programmatic Generation Strategy

```python
def create_room_brush_set(min_x, min_y, min_z, max_x, max_y, max_z, wall_thickness=8):
    """
    Generate 6 brushes forming a sealed room
    Each brush is a slab (floor, ceiling, 4 walls)
    """
    brushes = []
    
    # Floor slab: full XY extent, z from min_z to min_z+thickness
    floor = create_slab(
        min_x, min_y, min_z,
        max_x, max_y, min_z + wall_thickness,
        texture_top="base_floor/concrete",
        texture_others="common/caulk"
    )
    brushes.append(floor)
    
    # Ceiling slab: full XY extent, z from max_z-thickness to max_z
    ceiling = create_slab(
        min_x, min_y, max_z - wall_thickness,
        max_x, max_y, max_z,
        texture_bottom="base_floor/concrete",
        texture_others="common/caulk"
    )
    brushes.append(ceiling)
    
    # 4 wall slabs (each extends full height between floor and ceiling)
    # Left wall: x from min_x to min_x+thickness
    # Right wall: x from max_x-thickness to max_x
    # Back wall: y from min_y to min_y+thickness
    # Front wall: y from max_y-thickness to max_y
    
    return brushes
```

## Debugging Tips

1. **Start simple**: Single sealed box before complex geometry
2. **Use GtkRadiant**: Visual validation of generated maps
3. **Check q3map2 output**: Look for "0 surfaces" = bad geometry
4. **Verify bounds**: Should show actual coordinates, not 99999
5. **Compile incrementally**: Add one feature at a time
6. **Keep textures extracted**: Faster compilation than pak lookups

## CSG Operations

### Make Hollow

Converts solid brush into 6 wall brushes:
1. Calculate brush bounds
2. Create 6 slabs (thickness = grid size)
3. Each slab touches outer bounds, leaves hollow center
4. Delete original brush

### Subtract

Removes one brush's volume from others:
1. For each face of cutting brush
2. Split target brush by that plane
3. Keep only fragments outside cutting volume
4. Merge fragments where possible

### Merge

Combines multiple brushes into one:
1. Verify convex hull is valid
2. Remove internal faces
3. Keep only outer surfaces
4. Fails if result would be concave

## Integration with World Model System

```python
# Entity in spatial database → Box brush in map
entity = world_model.get_entity(entity_id)

if entity.primitive_type == PrimitiveType.BOX:
    # Convert meters to Quake units (32 units = 1 meter)
    min_q3 = entity.position - entity.dimensions / 2 * 32
    max_q3 = entity.position + entity.dimensions / 2 * 32
    
    # Generate brush set
    brushes = create_room_brush_set(
        min_q3.x, min_q3.y, min_q3.z,
        max_q3.x, max_q3.y, max_q3.z
    )
    
    # Add to map file
    for brush in brushes:
        map_file.add_brush(brush)
```

## References

- GtkRadiant source: `radiant/csg.cpp`
- Brush functions: `radiant/brush.cpp`
- q3map2 compiler: `tools/quake3/q3map2/`
- BobToolz examples: `contrib/bobtoolz/shapes.cpp`

## Summary

**Remember:**
1. Brushes are convex volumes defined by plane intersections
2. Each plane needs 3 points that lie on that plane
3. Neighboring brushes share planes for perfect alignment
4. One unique plane per brush defines its thickness
5. Use integer coordinates on 8-unit grid
6. Visible faces get real textures, hidden faces get `common/caulk`
7. Test simple cases first, add complexity incrementally
