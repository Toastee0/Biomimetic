/*
 * q3merge - Merge Quake 3 .map files
 * Combines structure (worldspawn brushes) with entities
 */

#include "../lib/q3map_common.h"
#include <getopt.h>

typedef enum { INIT, IN_WORLDSPAWN, IN_BRUSH, OUTSIDE } ParseState;

void copy_worldspawn_brushes(FILE *in, FILE *out);
void copy_entities_only(FILE *in, FILE *out);

int main(int argc, char **argv) {
    char *base_file = NULL;
    char *entity_file = NULL;
    char *output_file = NULL;
    FILE *base, *entities, *output;
    int opt;

    // Parse args
    while ((opt = getopt(argc, argv, "b:e:o:h")) != -1) {
        switch (opt) {
            case 'b': base_file = optarg; break;
            case 'e': entity_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 'h':
                printf("Usage: %s -b base.map -e entities.map -o output.map\n", argv[0]);
                printf("Or: %s base.map entities.map -o output.map\n", argv[0]);
                return 0;
            default: return 1;
        }
    }

    // Handle positional args
    if (!base_file && optind < argc) base_file = argv[optind++];
    if (!entity_file && optind < argc) entity_file = argv[optind++];

    if (!base_file || !output_file) {
        fprintf(stderr, "Error: Need base file and output file\n");
        return 1;
    }

    // Open files
    output = fopen(output_file, "w");
    if (!output) {
        fprintf(stderr, "Error: Cannot create %s\n", output_file);
        return 1;
    }

    fprintf(output, "// Merged map: %s + %s\n\n", base_file, entity_file ? entity_file : "");
    fprintf(output, "// entity 0\n");
    fprintf(output, "{\n\"classname\" \"worldspawn\"\n");

    // Copy structure brushes
    base = fopen(base_file, "r");
    if (base) {
        copy_worldspawn_brushes(base, output);
        fclose(base);
    }

    fprintf(output, "}\n\n");

    // Copy entities
    if (entity_file) {
        entities = fopen(entity_file, "r");
        if (entities) {
            copy_entities_only(entities, output);
            fclose(entities);
        }
    }

    fclose(output);
    fprintf(stderr, "Merged → %s\n", output_file);
    return 0;
}

void copy_worldspawn_brushes(FILE *in, FILE *out) {
    char line[1024];
    ParseState state = INIT;
    int depth = 0;
    int brush_num = 0;
    int in_brush_comment = 0;

    while (fgets(line, sizeof(line), in)) {
        // Copy brush comments or add them
        if (strstr(line, "// brush")) {
            fprintf(out, "// brush %d\n", brush_num++);
            in_brush_comment = 1;
            continue;
        }

        switch (state) {
            case INIT:
                if (strchr(line, '{')) state = IN_WORLDSPAWN;
                break;
            case IN_WORLDSPAWN:
                if (strchr(line, '{')) {
                    state = IN_BRUSH;
                    depth = 1;
                    // Add brush comment if not already present
                    if (!in_brush_comment) {
                        fprintf(out, "// brush %d\n", brush_num++);
                    }
                    in_brush_comment = 0;
                    fprintf(out, "%s", line);
                } else if (strchr(line, '}')) {
                    return; // Done
                } else if (line[0] != '"' && line[0] != '/') {
                    fprintf(out, "%s", line);
                }
                break;
            case IN_BRUSH:
                fprintf(out, "%s", line);
                if (strchr(line, '{')) depth++;
                else if (strchr(line, '}')) {
                    depth--;
                    if (depth == 0) state = IN_WORLDSPAWN;
                }
                break;
            default: break;
        }
    }
}

void copy_entities_only(FILE *in, FILE *out) {
    char line[1024];
    int in_entity = 0, depth = 0;
    int entity_num = 1; // Start at 1 (worldspawn is 0)
    int need_entity_comment = 0;

    while (fgets(line, sizeof(line), in)) {
        // Skip brush comments in entity files
        if (strstr(line, "// brush")) continue;

        // Handle entity comments
        if (strstr(line, "// entity")) continue;

        if (!in_entity && strchr(line, '{')) {
            // Check if not worldspawn
            long pos = ftell(in);
            char next[1024];
            int is_worldspawn = 0;
            if (fgets(next, sizeof(next), in) && strstr(next, "worldspawn"))
                is_worldspawn = 1;
            fseek(in, pos, SEEK_SET);

            if (!is_worldspawn) {
                in_entity = 1;
                depth = 1;
                fprintf(out, "// entity %d\n", entity_num++);
                fprintf(out, "%s", line);
            }
        } else if (in_entity) {
            fprintf(out, "%s", line);
            if (strchr(line, '{')) depth++;
            else if (strchr(line, '}')) {
                depth--;
                if (depth == 0) {
                    in_entity = 0;
                    fprintf(out, "\n");
                }
            }
        }
    }
}
