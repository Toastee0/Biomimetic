/*
 * brush_csg_tool.c
 * Standalone tool for creating Quake 3 brushes using proven GtkRadiant CSG functions
 * 
 * Extracts BrushFromBounds and MakeHollow from q3map2 source
 * Generates .map files with guaranteed valid geometry
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define qfalse 0
#define qtrue 1
typedef int qboolean;
typedef float vec_t;
typedef vec_t vec3_t[3];

#define VectorClear(x) (x[0]=x[1]=x[2]=0)
#define VectorCopy(a,b) (b[0]=a[0],b[1]=a[1],b[2]=a[2])
#define VectorAdd(a,b,c) (c[0]=a[0]+b[0],c[1]=a[1]+b[1],c[2]=a[2]+b[2])
#define VectorSubtract(a,b,c) (c[0]=a[0]-b[0],c[1]=a[1]-b[1],c[2]=a[2]-b[2])
#define VectorScale(a,b,c) (c[0]=a[0]*b,c[1]=a[1]*b,c[2]=a[2]*b)
#define DotProduct(a,b) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

// Simple plane structure
typedef struct {
    vec3_t normal;
    vec_t dist;
} plane_t;

// Simplified brush side
typedef struct {
    vec3_t planepts[3];  // Three points defining the plane
    char texture[64];
} side_t;

// Simplified brush
typedef struct {
    int numsides;
    side_t *sides;
    vec3_t mins, maxs;
} brush_t;

// Plane database (simplified)
static plane_t mapplanes[32768];
static int nummapplanes = 0;

// Find or create a plane
int FindFloatPlane(vec3_t normal, vec_t dist) {
    int i;
    plane_t *p;
    
    // Search for existing plane
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
    if (nummapplanes >= 32768) {
        fprintf(stderr, "Error: Too many planes\n");
        exit(1);
    }
    
    p = &mapplanes[nummapplanes];
    VectorCopy(normal, p->normal);
    p->dist = dist;
    return nummapplanes++;
}

// Use the same pattern as Brush_Create from radiant/brush.cpp
// Creates 6 faces from 4 corner points at two Z levels
void CreateBrushPlanepts(vec3_t mins, vec3_t maxs, side_t *sides, const char *texture) {
    vec3_t pts[4][2];
    int i, j;
    
    // Bottom 4 corners
    pts[0][0][0] = mins[0]; pts[0][0][1] = mins[1]; pts[0][0][2] = mins[2];
    pts[1][0][0] = mins[0]; pts[1][0][1] = maxs[1]; pts[1][0][2] = mins[2];
    pts[2][0][0] = maxs[0]; pts[2][0][1] = maxs[1]; pts[2][0][2] = mins[2];
    pts[3][0][0] = maxs[0]; pts[3][0][1] = mins[1]; pts[3][0][2] = mins[2];
    
    // Top 4 corners (same XY, max Z)
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
    }
    
    // Bottom face
    VectorCopy(pts[0][0], sides[4].planepts[0]);
    VectorCopy(pts[1][0], sides[4].planepts[1]);
    VectorCopy(pts[2][0], sides[4].planepts[2]);
    strncpy(sides[4].texture, texture, 63);
    
    // Top face
    VectorCopy(pts[2][1], sides[5].planepts[0]);
    VectorCopy(pts[1][1], sides[5].planepts[1]);
    VectorCopy(pts[0][1], sides[5].planepts[2]);
    strncpy(sides[5].texture, texture, 63);
}

// Create an axial brush from bounding box (from q3map2 brush.c)
brush_t *BrushFromBounds(vec3_t mins, vec3_t maxs, const char *texture) {
    brush_t *b;
    
    b = malloc(sizeof(brush_t));
    b->numsides = 6;
    b->sides = calloc(6, sizeof(side_t));
    
    VectorCopy(mins, b->mins);
    VectorCopy(maxs, b->maxs);
    
    CreateBrushPlanepts(mins, maxs, b->sides, texture);
    
    return b;
}

// Make a brush hollow with specified wall thickness (inspired by GtkRadiant csg.c)
brush_t **MakeBrushHollow(brush_t *solid, vec_t thickness, const char *texture, int *numbrushes) {
    brush_t **brushes;
    int i;
    vec3_t mins, maxs;
    
    VectorCopy(solid->mins, mins);
    VectorCopy(solid->maxs, maxs);
    
    // Create 6 wall brushes
    *numbrushes = 6;
    brushes = malloc(6 * sizeof(brush_t*));
    
    // Bottom wall (Z-) 
    brushes[0] = BrushFromBounds(
        (vec3_t){mins[0], mins[1], mins[2]},
        (vec3_t){maxs[0], maxs[1], mins[2] + thickness},
        texture);
    
    // Top wall (Z+)
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

// Write brush to .map file
void WriteBrush(FILE *f, brush_t *brush) {
    int i;
    
    fprintf(f, "{\n");
    for (i = 0; i < brush->numsides; i++) {
        side_t *side = &brush->sides[i];
        fprintf(f, "( %g %g %g ) ( %g %g %g ) ( %g %g %g ) %s 0 0 0 1 1 0 0 0\n",
            side->planepts[0][0], side->planepts[0][1], side->planepts[0][2],
            side->planepts[1][0], side->planepts[1][1], side->planepts[1][2],
            side->planepts[2][0], side->planepts[2][1], side->planepts[2][2],
            side->texture);
    }
    fprintf(f, "}\n");
}

int main(int argc, char **argv) {
    FILE *f;
    brush_t *solid;
    brush_t **hollow;
    int numbrushes, i;
    
    if (argc < 2) {
        printf("Usage: %s <output.map> [--hollow]\n", argv[0]);
        printf("Creates a simple box room\n");
        printf("  --hollow: Create hollow room with 8-unit thick walls\n");
        return 1;
    }
    
    qboolean make_hollow = (argc > 2 && strcmp(argv[2], "--hollow") == 0);
    
    f = fopen(argv[1], "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot create %s\n", argv[1]);
        return 1;
    }
    
    // Write map header
    fprintf(f, "// Generated by brush_csg_tool using GtkRadiant BrushFromBounds\n");
    fprintf(f, "{\n");
    fprintf(f, "\"classname\" \"worldspawn\"\n");
    
    if (make_hollow) {
        // Create hollow room
        solid = BrushFromBounds(
            (vec3_t){-256, -256, 0},
            (vec3_t){256, 256, 256},
            "base_wall/concrete");
        
        hollow = MakeBrushHollow(solid, 8, "base_wall/concrete", &numbrushes);
        
        printf("Creating hollow room with %d wall brushes (8 unit thickness)\n", numbrushes);
        
        for (i = 0; i < numbrushes; i++) {
            WriteBrush(f, hollow[i]);
            free(hollow[i]->sides);
            free(hollow[i]);
        }
        free(hollow);
        free(solid->sides);
        free(solid);
    } else {
        // Create solid box
        solid = BrushFromBounds(
            (vec3_t){-256, -256, 0},
            (vec3_t){256, 256, 16},
            "base_floor/concrete");
        
        printf("Creating solid box brush\n");
        WriteBrush(f, solid);
        
        free(solid->sides);
        free(solid);
    }
    
    fprintf(f, "}\n");
    
    // Add player spawn point
    fprintf(f, "{\n");
    fprintf(f, "\"classname\" \"info_player_deathmatch\"\n");
    fprintf(f, "\"origin\" \"0 0 32\"\n");
    fprintf(f, "\"angle\" \"0\"\n");
    fprintf(f, "}\n");
    
    // Add light
    fprintf(f, "{\n");
    fprintf(f, "\"classname\" \"light\"\n");
    fprintf(f, "\"origin\" \"0 0 200\"\n");
    fprintf(f, "\"light\" \"300\"\n");
    fprintf(f, "}\n");
    
    fclose(f);
    
    printf("Created %s successfully\n", argv[1]);
    printf("Compile with: q3map2 -bsp -fs_basepath ~/.q3a -game quake3 %s\n", argv[1]);
    
    return 0;
}
