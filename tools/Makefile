# Makefile for Quake 3 CLI Tools
# Fast map generation for BioMimetic AI spatial twin system

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Ilib -I../external/cJSON
LDFLAGS = -lm

# Directories
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
CJSON_DIR = ../external/cJSON

# Source files
LIB_SOURCES = $(LIB_DIR)/q3map_common.c
CJSON_SOURCES = $(CJSON_DIR)/cJSON.c

# Object files
LIB_OBJECTS = $(BUILD_DIR)/q3map_common.o
CJSON_OBJECTS = $(BUILD_DIR)/cJSON.o

# Tools (output to bin/)
TOOLS = $(BIN_DIR)/q3entity $(BIN_DIR)/q3merge
SCRIPTS = q3reload

# Targets
.PHONY: all clean install test help

all: $(BUILD_DIR) $(BIN_DIR) $(TOOLS)
	@echo ""
	@echo "=== Build Complete ==="
	@echo "Tools built in $(BIN_DIR)/"
	@echo "Run 'make install' to install to /usr/local/bin"
	@echo "Run 'make test' to run basic tests"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Build shared library objects
$(BUILD_DIR)/q3map_common.o: $(LIB_SOURCES) $(LIB_DIR)/q3map_common.h
	@echo "Compiling shared library..."
	$(CC) $(CFLAGS) -c $(LIB_SOURCES) -o $@

$(BUILD_DIR)/cJSON.o: $(CJSON_SOURCES) $(CJSON_DIR)/cJSON.h
	@echo "Compiling cJSON..."
	$(CC) $(CFLAGS) -c $(CJSON_SOURCES) -o $@

# Build q3entity (requires cJSON)
$(BIN_DIR)/q3entity: $(BUILD_DIR)/q3entity.o $(LIB_OBJECTS) $(CJSON_OBJECTS)
	@echo "Linking q3entity..."
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓ q3entity built"

$(BUILD_DIR)/q3entity.o: q3entity/main.c $(LIB_DIR)/q3map_common.h
	@echo "Compiling q3entity..."
	$(CC) $(CFLAGS) -c $< -o $@

# Build q3merge
$(BIN_DIR)/q3merge: $(BUILD_DIR)/q3merge.o $(LIB_OBJECTS)
	@echo "Linking q3merge..."
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓ q3merge built"

$(BUILD_DIR)/q3merge.o: q3merge/main.c $(LIB_DIR)/q3map_common.h
	@echo "Compiling q3merge..."
	$(CC) $(CFLAGS) -c $< -o $@

# Install to /usr/local/bin
install: all
	@echo "Installing tools to /usr/local/bin..."
	@sudo install -m 755 $(BIN_DIR)/q3entity /usr/local/bin/q3entity
	@sudo install -m 755 $(BIN_DIR)/q3merge /usr/local/bin/q3merge
	@sudo install -m 755 $(SCRIPTS) /usr/local/bin/
	@echo "✓ Installation complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Clean complete"

# Basic tests
test: all
	@echo ""
	@echo "=== Running Basic Tests ==="
	@echo ""
	@echo "[Test 1] q3entity --help"
	@./q3entity --help || true
	@echo ""
	@echo "[Test 2] q3merge --help"
	@./q3merge --help || true
	@echo ""
	@echo "[Test 3] Creating test entity JSON..."
	@mkdir -p test
	@echo '{"entities":[{"id":"test1","label":"test_box","entity_type":"OBJECT","primitive":"BOX","position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1],"importance":0.9}]}' > test/test_entities.json
	@echo ""
	@echo "[Test 4] Converting JSON to .map entities..."
	@$(BIN_DIR)/q3entity test/test_entities.json -o test/test_entities.map
	@if [ -f test/test_entities.map ]; then \
		echo "✓ Test entities.map created"; \
		head -20 test/test_entities.map; \
	else \
		echo "✗ Failed to create test entities.map"; \
	fi
	@echo ""
	@echo "=== Tests Complete ==="

# Help
help:
	@echo "Quake 3 CLI Tools - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build all tools"
	@echo "  make install   - Install tools to /usr/local/bin (requires sudo)"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make test      - Run basic tests"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Tools:"
	@echo "  q3entity   - Convert entity JSON to .map entities"
	@echo "  q3merge    - Merge structure and entity .map files"
	@echo "  q3reload   - Hot reload map in Quake 3 engine"
	@echo ""
	@echo "Example workflow:"
	@echo "  make"
	@echo "  make test"
	@echo "  make install"
	@echo "  q3entity entities.json -o entities.map"
	@echo "  q3merge structure.map entities.map -o world.map"
	@echo "  q3reload world.map --entities-only"

# Dependency tracking
-include $(BUILD_DIR)/*.d
