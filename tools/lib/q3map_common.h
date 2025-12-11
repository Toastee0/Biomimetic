/*
 * q3map_common.h
 * Shared data structures and functions for Quake 3 map generation tools
 *
 * Extracted from brush_csg_tool.c for reuse across all CLI tools
 */

#ifndef Q3MAP_COMMON_H
#define Q3MAP_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Type definitions
#define qfalse 0
#define qtrue 1
typedef int qboolean;
typedef float vec_t;
typedef vec_t vec3_t[3];

// Vector math macros
#define VectorClear(x) (x[0]=x[1]=x[2]=0)
#define VectorCopy(a,b) (b[0]=a[0],b[1]=a[1],b[2]=a[2])
#define VectorAdd(a,b,c) (c[0]=a[0]+b[0],c[1]=a[1]+b[1],c[2]=a[2]+b[2])
#define VectorSubtract(a,b,c) (c[0]=a[0]-b[0],c[1]=a[1]-b[1],c[2]=a[2]-b[2])
#define VectorScale(a,b,c) (c[0]=a[0]*b,c[1]=a[1]*b,c[2]=a[2]*b)
#define DotProduct(a,b) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
#define VectorLength(a) sqrt(DotProduct(a,a))

// Plane structure
typedef struct {
    vec3_t normal;
    vec_t dist;
} plane_t;

// Brush side (face)
typedef struct {
    vec3_t planepts[3];  // Three points defining the plane
    char texture[64];
} side_t;

// Brush (convex polyhedron)
typedef struct {
    int numsides;
    side_t *sides;
    vec3_t mins, maxs;
} brush_t;

// Entity structure for JSON parsing
typedef struct {
    char id[128];
    char label[128];
    char entity_type[32];
    char primitive[32];
    vec3_t position;
    vec3_t rotation;
    vec3_t scale;
    vec3_t velocity;
    float importance;
} entity_t;

// Plane database
#define MAX_MAP_PLANES 32768
extern plane_t mapplanes[MAX_MAP_PLANES];
extern int nummapplanes;

// Core functions
int FindFloatPlane(vec3_t normal, vec_t dist);
void CreateBrushPlanepts(vec3_t mins, vec3_t maxs, side_t *sides, const char *texture);
brush_t *BrushFromBounds(vec3_t mins, vec3_t maxs, const char *texture);
brush_t **MakeBrushHollow(brush_t *solid, vec_t thickness, const char *texture, int *numbrushes);
void WriteBrush(FILE *f, brush_t *brush);
void WriteMapHeader(FILE *f, const char *comment);
void WriteMapFooter(FILE *f);
void FreeBrush(brush_t *brush);

// Coordinate conversion
void WorldToQuake(vec3_t world, vec3_t quake);
void QuakeToWorld(vec3_t quake, vec3_t world);

// Importance to color mapping
void ImportanceToColor(float importance, float *r, float *g, float *b);

// Entity helpers
const char *GetModelForPrimitive(const char *primitive);
const char *GetModelForEntityType(const char *entity_type);

#endif // Q3MAP_COMMON_H
