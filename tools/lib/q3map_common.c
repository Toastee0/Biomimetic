/*
 * q3map_common.c
 * Shared implementation for Quake 3 map generation tools
 */

#include "q3map_common.h"

// Global plane database
plane_t mapplanes[MAX_MAP_PLANES];
int nummapplanes = 0;

// Find or create a plane in the global database
int FindFloatPlane(vec3_t normal, vec_t dist) {
    int i;
    plane_t *p;

    // Search for existing plane (with tolerance)
    for (i = 0; i < nummapplanes; i++) {
        p = &mapplanes[i];
        if (fabs(p->normal[0] - normal[0]) < 0.001 &&
            fabs(p->normal[1] - normal[1]) < 0.001 &&
            fabs(p->normal[2] - normal[2]) < 0.001 &&
            fabs(p->dist - dist) < 0.01) {
            return i;
        }
    }

    // Create new plane
    if (nummapplanes >= MAX_MAP_PLANES) {
        fprintf(stderr, "Error: Too many planes (max %d)\n", MAX_MAP_PLANES);
        exit(1);
    }

    p = &mapplanes[nummapplanes];
    VectorCopy(normal, p->normal);
    p->dist = dist;
    return nummapplanes++;
}

// Creates 6 faces from 4 corner points at two Z levels
// Pattern from GtkRadiant Brush_Create
void CreateBrushPlanepts(vec3_t mins, vec3_t maxs, side_t *sides, const char *texture) {
    vec3_t pts[4][2];
    int i, j;

    // Bottom 4 corners (Z = mins[2])
    pts[0][0][0] = mins[0]; pts[0][0][1] = mins[1]; pts[0][0][2] = mins[2];
    pts[1][0][0] = mins[0]; pts[1][0][1] = maxs[1]; pts[1][0][2] = mins[2];
    pts[2][0][0] = maxs[0]; pts[2][0][1] = maxs[1]; pts[2][0][2] = mins[2];
    pts[3][0][0] = maxs[0]; pts[3][0][1] = mins[1]; pts[3][0][2] = mins[2];

    // Top 4 corners (same XY, Z = maxs[2])
    for (i = 0; i < 4; i++) {
        pts[i][1][0] = pts[i][0][0];
        pts[i][1][1] = pts[i][0][1];
        pts[i][1][2] = maxs[2];
    }

    // Create 4 vertical faces
    for (i = 0; i < 4; i++) {
        j = (i + 1) % 4;
        VectorCopy(pts[j][1], sides[i].planepts[0]);
        VectorCopy(pts[i][1], sides[i].planepts[1]);
        VectorCopy(pts[i][0], sides[i].planepts[2]);
        strncpy(sides[i].texture, texture, 63);
        sides[i].texture[63] = '\0';
    }

    // Bottom face (Z-)
    VectorCopy(pts[0][0], sides[4].planepts[0]);
    VectorCopy(pts[1][0], sides[4].planepts[1]);
    VectorCopy(pts[2][0], sides[4].planepts[2]);
    strncpy(sides[4].texture, texture, 63);
    sides[4].texture[63] = '\0';

    // Top face (Z+)
    VectorCopy(pts[2][1], sides[5].planepts[0]);
    VectorCopy(pts[1][1], sides[5].planepts[1]);
    VectorCopy(pts[0][1], sides[5].planepts[2]);
    strncpy(sides[5].texture, texture, 63);
    sides[5].texture[63] = '\0';
}

// Create an axial brush from bounding box
brush_t *BrushFromBounds(vec3_t mins, vec3_t maxs, const char *texture) {
    brush_t *b;

    b = malloc(sizeof(brush_t));
    if (!b) {
        fprintf(stderr, "Error: Failed to allocate brush\n");
        exit(1);
    }

    b->numsides = 6;
    b->sides = calloc(6, sizeof(side_t));
    if (!b->sides) {
        fprintf(stderr, "Error: Failed to allocate brush sides\n");
        exit(1);
    }

    VectorCopy(mins, b->mins);
    VectorCopy(maxs, b->maxs);

    CreateBrushPlanepts(mins, maxs, b->sides, texture);

    return b;
}

// Make a brush hollow with specified wall thickness
// Inspired by GtkRadiant CSG_MakeHollow
brush_t **MakeBrushHollow(brush_t *solid, vec_t thickness, const char *texture, int *numbrushes) {
    brush_t **brushes;
    vec3_t mins, maxs;

    VectorCopy(solid->mins, mins);
    VectorCopy(solid->maxs, maxs);

    // Create 6 wall brushes (floor, ceiling, 4 walls)
    *numbrushes = 6;
    brushes = malloc(6 * sizeof(brush_t*));
    if (!brushes) {
        fprintf(stderr, "Error: Failed to allocate hollow brushes\n");
        exit(1);
    }

    // Bottom wall (floor)
    brushes[0] = BrushFromBounds(
        (vec3_t){mins[0], mins[1], mins[2]},
        (vec3_t){maxs[0], maxs[1], mins[2] + thickness},
        texture);

    // Top wall (ceiling)
    brushes[1] = BrushFromBounds(
        (vec3_t){mins[0], mins[1], maxs[2] - thickness},
        (vec3_t){maxs[0], maxs[1], maxs[2]},
        texture);

    // Left wall (X-)
    brushes[2] = BrushFromBounds(
        (vec3_t){mins[0], mins[1], mins[2] + thickness},
        (vec3_t){mins[0] + thickness, maxs[1], maxs[2] - thickness},
        texture);

    // Right wall (X+)
    brushes[3] = BrushFromBounds(
        (vec3_t){maxs[0] - thickness, mins[1], mins[2] + thickness},
        (vec3_t){maxs[0], maxs[1], maxs[2] - thickness},
        texture);

    // Back wall (Y-)
    brushes[4] = BrushFromBounds(
        (vec3_t){mins[0] + thickness, mins[1], mins[2] + thickness},
        (vec3_t){maxs[0] - thickness, mins[1] + thickness, maxs[2] - thickness},
        texture);

    // Front wall (Y+)
    brushes[5] = BrushFromBounds(
        (vec3_t){mins[0] + thickness, maxs[1] - thickness, mins[2] + thickness},
        (vec3_t){maxs[0] - thickness, maxs[1], maxs[2] - thickness},
        texture);

    return brushes;
}

// Write brush to .map file in standard Quake 3 format
void WriteBrush(FILE *f, brush_t *brush) {
    int i;

    fprintf(f, "{\n");
    for (i = 0; i < brush->numsides; i++) {
        side_t *side = &brush->sides[i];
        // Format: ( x y z ) ( x y z ) ( x y z ) texture xoff yoff rot xscale yscale contentflags surfaceflags value
        fprintf(f, "( %g %g %g ) ( %g %g %g ) ( %g %g %g ) %s 0 0 0 1 1 0 0 0\n",
            side->planepts[0][0], side->planepts[0][1], side->planepts[0][2],
            side->planepts[1][0], side->planepts[1][1], side->planepts[1][2],
            side->planepts[2][0], side->planepts[2][1], side->planepts[2][2],
            side->texture);
    }
    fprintf(f, "}\n");
}

// Write .map file header
void WriteMapHeader(FILE *f, const char *comment) {
    fprintf(f, "// %s\n", comment);
    fprintf(f, "{\n");
    fprintf(f, "\"classname\" \"worldspawn\"\n");
}

// Write .map file footer
void WriteMapFooter(FILE *f) {
    fprintf(f, "}\n");
}

// Free brush memory
void FreeBrush(brush_t *brush) {
    if (brush) {
        if (brush->sides) {
            free(brush->sides);
        }
        free(brush);
    }
}

// Convert World Model coordinates to Quake 3 coordinates
// World Model: Y-up, meters
// Quake 3: Z-up, 32 units = 1 meter
void WorldToQuake(vec3_t world, vec3_t quake) {
    quake[0] = world[0] * 32.0f;  // X unchanged
    quake[1] = world[2] * 32.0f;  // Y ← Z (swap Y and Z)
    quake[2] = world[1] * 32.0f;  // Z ← Y
}

// Convert Quake 3 coordinates to World Model coordinates
void QuakeToWorld(vec3_t quake, vec3_t world) {
    world[0] = quake[0] / 32.0f;  // X unchanged
    world[1] = quake[2] / 32.0f;  // Y ← Z (swap back)
    world[2] = quake[1] / 32.0f;  // Z ← Y
}

// Map importance (0-1) to RGB color
// High importance = Red, Medium = Yellow, Low = Green, Very low = Gray
void ImportanceToColor(float importance, float *r, float *g, float *b) {
    if (importance >= 0.8f) {
        // High: Red
        *r = 1.0f; *g = 0.2f; *b = 0.2f;
    } else if (importance >= 0.5f) {
        // Medium: Yellow
        *r = 1.0f; *g = 1.0f; *b = 0.2f;
    } else if (importance >= 0.2f) {
        // Low: Green
        *r = 0.2f; *g = 1.0f; *b = 0.2f;
    } else {
        // Very low: Gray
        *r = 0.5f; *g = 0.5f; *b = 0.5f;
    }
}

// Get Quake 3 model path for primitive type
const char *GetModelForPrimitive(const char *primitive) {
    if (strcmp(primitive, "BOX") == 0) {
        return "models/mapobjects/box.md3";
    } else if (strcmp(primitive, "SPHERE") == 0) {
        return "models/mapobjects/sphere.md3";
    } else if (strcmp(primitive, "CYLINDER") == 0) {
        return "models/mapobjects/cylinder.md3";
    } else if (strcmp(primitive, "CAPSULE") == 0) {
        return "models/players/sarge/upper.md3";  // Use player model for human capsules
    } else {
        return "models/mapobjects/box.md3";  // Default
    }
}

// Get Quake 3 model path for entity type
const char *GetModelForEntityType(const char *entity_type) {
    if (strcmp(entity_type, "HUMAN") == 0) {
        return "models/players/sarge/default.md3";
    } else if (strcmp(entity_type, "ROBOT") == 0) {
        return "models/custom/robot.md3";
    } else if (strcmp(entity_type, "FURNITURE") == 0) {
        return "models/mapobjects/furniture/desk.md3";
    } else if (strcmp(entity_type, "APPLIANCE") == 0) {
        return "models/mapobjects/appliance.md3";
    } else if (strcmp(entity_type, "OBJECT") == 0) {
        return "models/mapobjects/generic_box.md3";
    } else if (strcmp(entity_type, "ZONE") == 0) {
        return "models/mapobjects/zone_marker.md3";
    } else {
        return "models/mapobjects/box.md3";  // Default
    }
}
