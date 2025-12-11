"""
Map generator for World Model visualization.

Generates Quake 3 .map files from World Model spatial database,
including AI observer camera and entity representations.
"""

import math
from typing import List, Tuple
from pathlib import Path

from ..entity import Entity, EntityType, PrimitiveType
from ..world_model import WorldModel


class MapGenerator:
    """
    Generate Quake 3 map files from World Model.
    
    Creates playable .map files with:
    - AI observer camera (shows AI's viewpoint)
    - Entity representations as brushes/models
    - Importance-based lighting
    - Navigation hints
    """
    
    QUAKE_UNIT = 32  # 1 meter = 32 Quake units
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
    
    def generate(self, output_path: str):
        """Generate complete .map file."""
        lines = []
        
        # Header
        lines.extend(self._generate_header())
        
        # Worldspawn (contains brushes like floor, walls)
        lines.append("{")
        lines.append('"classname" "worldspawn"')
        lines.append('"message" "BioMimetic AI World Model"')
        lines.append('"_color" "0.8 0.8 0.9"')
        lines.append("")
        
        # Generate floor and basic structure
        lines.extend(self._generate_floor())
        lines.extend(self._generate_bounding_walls())
        
        lines.append("}")
        lines.append("")
        
        # AI Observer Camera (main viewpoint)
        lines.extend(self._generate_ai_camera())
        
        # Player spawn points
        lines.extend(self._generate_spawn_points())
        
        # Light sources based on importance
        lines.extend(self._generate_importance_lights())
        
        # Generate entity representations
        entities = self.world_model.db.get_all_entities()
        for entity in entities:
            lines.extend(self._generate_entity(entity))
            lines.append("")
        
        # Write file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Generated map: {output_path}")
        print(f"  Entities: {len(entities)}")
        print(f"  Origin: {self.world_model.origin}")
    
    def _generate_header(self) -> List[str]:
        """Generate map file header."""
        return [
            "// BioMimetic AI World Model",
            "// Generated from spatial database",
            f"// Origin: {self.world_model.origin}",
            "// Entity count: {}".format(len(self.world_model.db.get_all_entities())),
            ""
        ]
    
    def _generate_floor(self) -> List[str]:
        """Generate floor brush (16 units thick)."""
        size = 1024  # Room boundary
        return [
            "// Floor",
            "{",
            f'( {-size} {-size} 0 ) ( {size} {-size} 0 ) ( {size} {size} 0 ) textures/base_floor/techfloor2 0 0 0 0.5 0.5',
            f'( {-size} {-size} -16 ) ( {-size} {size} -16 ) ( {size} {size} -16 ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size} -16 ) ( {-size} {-size} 0 ) ( {-size} {size} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {-size} -16 ) ( {size} {size} -16 ) ( {size} {size} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size} -16 ) ( {size} {-size} -16 ) ( {size} {-size} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {size} -16 ) ( {-size} {size} 0 ) ( {size} {size} 0 ) common/caulk 0 0 0 0.5 0.5',
            "}",
            ""
        ]
    
    def _generate_bounding_walls(self) -> List[str]:
        """Generate 4 walls + ceiling to create sealed room."""
        size = 1024
        height = 256
        thickness = 16
        lines = ["// Walls and ceiling"]
        
        # North wall (y+)
        lines.extend([
            "{",
            f'( {-size} {size} {height} ) ( {size} {size} {height} ) ( {size} {size+thickness} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {-size} {size} 0 ) ( {-size} {size+thickness} 0 ) ( {size} {size+thickness} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {size} 0 ) ( {-size} {size} {height} ) ( {-size} {size+thickness} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {size} 0 ) ( {size} {size+thickness} 0 ) ( {size} {size+thickness} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {size} 0 ) ( {size} {size} 0 ) ( {size} {size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {-size} {size+thickness} 0 ) ( {-size} {size+thickness} {height} ) ( {size} {size+thickness} {height} ) common/caulk 0 0 0 0.5 0.5',
            "}",
            ""
        ])
        
        # South wall (y-)
        lines.extend([
            "{",
            f'( {-size} {-size-thickness} {height} ) ( {size} {-size-thickness} {height} ) ( {size} {-size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {-size} {-size-thickness} 0 ) ( {-size} {-size} 0 ) ( {size} {-size} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size-thickness} 0 ) ( {-size} {-size-thickness} {height} ) ( {-size} {-size} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {-size-thickness} 0 ) ( {size} {-size} 0 ) ( {size} {-size} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size-thickness} 0 ) ( {size} {-size-thickness} 0 ) ( {size} {-size-thickness} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size} 0 ) ( {-size} {-size} {height} ) ( {size} {-size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            "}",
            ""
        ])
        
        # East wall (x+)
        lines.extend([
            "{",
            f'( {size} {-size} {height} ) ( {size+thickness} {-size} {height} ) ( {size+thickness} {size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {size} {-size} 0 ) ( {size} {size} 0 ) ( {size+thickness} {size} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {-size} 0 ) ( {size} {-size} {height} ) ( {size} {size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {size+thickness} {-size} 0 ) ( {size+thickness} {size} 0 ) ( {size+thickness} {size} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {-size} 0 ) ( {size+thickness} {-size} 0 ) ( {size+thickness} {-size} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {size} 0 ) ( {size} {size} {height} ) ( {size+thickness} {size} {height} ) common/caulk 0 0 0 0.5 0.5',
            "}",
            ""
        ])
        
        # West wall (x-)
        lines.extend([
            "{",
            f'( {-size-thickness} {-size} {height} ) ( {-size} {-size} {height} ) ( {-size} {size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {-size-thickness} {-size} 0 ) ( {-size-thickness} {size} 0 ) ( {-size} {size} 0 ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size-thickness} {-size} 0 ) ( {-size-thickness} {-size} {height} ) ( {-size-thickness} {size} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size} 0 ) ( {-size} {size} 0 ) ( {-size} {size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {-size-thickness} {-size} 0 ) ( {-size} {-size} 0 ) ( {-size} {-size} {height} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size-thickness} {size} 0 ) ( {-size-thickness} {size} {height} ) ( {-size} {size} {height} ) common/caulk 0 0 0 0.5 0.5',
            "}",
            ""
        ])
        
        # Ceiling
        lines.extend([
            "{",
            f'( {-size} {-size} {height+thickness} ) ( {size} {-size} {height+thickness} ) ( {size} {size} {height+thickness} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size} {height} ) ( {-size} {size} {height} ) ( {size} {size} {height} ) textures/base_wall/basewall01 0 0 0 0.5 0.5',
            f'( {-size} {-size} {height} ) ( {-size} {-size} {height+thickness} ) ( {-size} {size} {height+thickness} ) common/caulk 0 0 0 0.5 0.5',
            f'( {size} {-size} {height} ) ( {size} {size} {height} ) ( {size} {size} {height+thickness} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {-size} {height} ) ( {size} {-size} {height} ) ( {size} {-size} {height+thickness} ) common/caulk 0 0 0 0.5 0.5',
            f'( {-size} {size} {height} ) ( {-size} {size} {height+thickness} ) ( {size} {size} {height+thickness} ) common/caulk 0 0 0 0.5 0.5',
            "}",
            ""
        ])
        
        return lines
    
    def _generate_brush(
        self, 
        position: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        texture: str = "common/caulk"
    ) -> List[str]:
        """Generate a simple box brush using standard Quake 3 plane format."""
        x, y, z = position
        w, d, h = [s / 2 for s in scale]  # Half-extents
        
        # Calculate box corners
        x0, x1 = x - w, x + w
        y0, y1 = y - d, y + d
        z0, z1 = z - h, z + h
        
        return [
            "{",
            # Top face (z+)
            f'( {x0} {y0} {z1} ) ( {x1} {y0} {z1} ) ( {x1} {y1} {z1} ) {texture} 0 0 0 0.5 0.5',
            # Bottom face (z-)
            f'( {x0} {y0} {z0} ) ( {x0} {y1} {z0} ) ( {x1} {y1} {z0} ) {texture} 0 0 0 0.5 0.5',
            # West face (x-)
            f'( {x0} {y0} {z0} ) ( {x0} {y0} {z1} ) ( {x0} {y1} {z1} ) {texture} 0 0 0 0.5 0.5',
            # East face (x+)
            f'( {x1} {y0} {z0} ) ( {x1} {y1} {z0} ) ( {x1} {y1} {z1} ) {texture} 0 0 0 0.5 0.5',
            # South face (y-)
            f'( {x0} {y0} {z0} ) ( {x1} {y0} {z0} ) ( {x1} {y0} {z1} ) {texture} 0 0 0 0.5 0.5',
            # North face (y+)
            f'( {x0} {y1} {z0} ) ( {x0} {y1} {z1} ) ( {x1} {y1} {z1} ) {texture} 0 0 0 0.5 0.5',
            "}",
        ]
    
    def _generate_ai_camera(self) -> List[str]:
        """
        Generate AI observer camera entity.
        
        This represents where the AI's attention/viewpoint currently is.
        """
        origin = self.world_model.origin
        x, y, z = [o * self.QUAKE_UNIT for o in origin]
        
        # Camera position (slightly above origin)
        cam_z = z + 64  # 2 meters above origin
        
        return [
            "// AI Observer Camera",
            "{",
            '"classname" "info_player_intermission"',
            f'"origin" "{x} {y} {cam_z}"',
            '"angle" "0"',
            '"target" "ai_focus"',
            '"notfree" "1"',
            '"notsingle" "1"',
            "}",
            "",
            "// AI focus point marker",
            "{",
            '"classname" "target_position"',
            '"targetname" "ai_focus"',
            f'"origin" "{x} {y} {z}"',
            "}",
            "",
            "// AI viewpoint indicator (visible model)",
            "{",
            '"classname" "misc_model"',
            '"model" "models/mapobjects/teleporter/energy.md3"',
            f'"origin" "{x} {y} {cam_z}"',
            '"modelscale" "2.0"',
            '"_color" "0.2 0.8 1.0"',
            '"notfree" "1"',
            '"notsingle" "1"',
            "}",
            ""
        ]
    
    def _generate_spawn_points(self) -> List[str]:
        """Generate player spawn points around the AI origin."""
        origin = self.world_model.origin
        x, y, z = [o * self.QUAKE_UNIT for o in origin]
        
        # Spawn points in a circle around origin
        spawns = []
        for i in range(8):
            angle = (360.0 / 8) * i
            rad = math.radians(angle)
            offset = 200  # About 6 meters away
            
            spawn_x = x + offset * math.cos(rad)
            spawn_y = y + offset * math.sin(rad)
            spawn_z = z + 40  # Slightly above floor
            
            spawns.extend([
                f"// Player spawn {i+1}",
                "{",
                '"classname" "info_player_deathmatch"',
                f'"origin" "{spawn_x:.1f} {spawn_y:.1f} {spawn_z:.1f}"',
                f'"angle" "{(angle + 180) % 360:.1f}"',  # Face toward center
                "}",
                ""
            ])
        
        return spawns
    
    def _generate_importance_lights(self) -> List[str]:
        """Generate lights based on entity importance clusters."""
        # Add ambient light at origin
        origin = self.world_model.origin
        x, y, z = [o * self.QUAKE_UNIT for o in origin]
        
        return [
            "// Ambient light at AI origin",
            "{",
            '"classname" "light"',
            f'"origin" "{x} {y} {z + 200}"',
            '"light" "300"',
            '"_color" "0.9 0.9 1.0"',
            "}",
            ""
        ]
    
    def _generate_entity(self, entity: Entity) -> List[str]:
        """Generate map representation of an entity."""
        # Convert World Model coords to Quake coords
        x, y, z = entity.position
        qx = x * self.QUAKE_UNIT
        qy = y * self.QUAKE_UNIT
        qz = z * self.QUAKE_UNIT
        
        # Scale
        sx, sy, sz = entity.scale
        qsx = sx * self.QUAKE_UNIT
        qsy = sy * self.QUAKE_UNIT
        qsz = sz * self.QUAKE_UNIT
        
        # Choose representation based on entity type
        if entity.entity_type == EntityType.HUMAN:
            return self._generate_human_entity(entity, (qx, qy, qz))
        elif entity.entity_type == EntityType.ROBOT:
            return self._generate_robot_entity(entity, (qx, qy, qz))
        elif entity.entity_type == EntityType.FURNITURE:
            return self._generate_furniture_entity(entity, (qx, qy, qz), (qsx, qsy, qsz))
        else:
            return self._generate_generic_entity(entity, (qx, qy, qz), (qsx, qsy, qsz))
    
    def _generate_human_entity(self, entity: Entity, pos: Tuple[float, float, float]) -> List[str]:
        """Generate human representation."""
        x, y, z = pos
        
        # Importance to color
        r, g, b = self._importance_to_color(entity.importance)
        
        return [
            f"// Human: {entity.label}",
            "{",
            '"classname" "misc_model"',
            '"model" "models/players/sarge/upper.md3"',
            f'"origin" "{x:.1f} {y:.1f} {z:.1f}"',
            '"modelscale" "1.0"',
            f'"_color" "{r:.2f} {g:.2f} {b:.2f}"',
            f'"message" "{entity.label}"',
            f'"importance" "{entity.importance:.2f}"',
            "}",
            "{",
            '"classname" "light"',
            f'"origin" "{x:.1f} {y:.1f} {z + 64:.1f}"',
            '"light" "150"',
            f'"_color" "{r:.2f} {g:.2f} {b:.2f}"',
            "}",
        ]
    
    def _generate_robot_entity(self, entity: Entity, pos: Tuple[float, float, float]) -> List[str]:
        """Generate robot representation."""
        x, y, z = pos
        r, g, b = self._importance_to_color(entity.importance)
        
        return [
            f"// Robot: {entity.label}",
            "{",
            '"classname" "misc_model"',
            '"model" "models/mapobjects/teleporter/widget.md3"',
            f'"origin" "{x:.1f} {y:.1f} {z:.1f}"',
            '"modelscale" "2.0"',
            f'"_color" "{r:.2f} {g:.2f} {b:.2f}"',
            f'"message" "{entity.label}"',
            "}",
        ]
    
    def _generate_furniture_entity(
        self, 
        entity: Entity, 
        pos: Tuple[float, float, float],
        scale: Tuple[float, float, float]
    ) -> List[str]:
        """Generate furniture as brush."""
        x, y, z = pos
        w, d, h = scale
        
        # Choose texture based on properties
        texture = "textures/base_wall/chrome_env"
        if entity.properties.get("material") == "wood":
            texture = "textures/base_trim/border11light"
        
        lines = [f"// Furniture: {entity.label}"]
        lines.append("{")
        lines.append('"classname" "func_group"')
        lines.extend(self._generate_brush((x, y, z), (w, d, h), texture))
        lines.append("}")
        
        return lines
    
    def _generate_generic_entity(
        self,
        entity: Entity,
        pos: Tuple[float, float, float],
        scale: Tuple[float, float, float]
    ) -> List[str]:
        """Generate generic entity representation."""
        x, y, z = pos
        r, g, b = self._importance_to_color(entity.importance)
        
        # Use different models based on primitive type
        model_map = {
            PrimitiveType.BOX: "models/mapobjects/barrel/barrel.md3",
            PrimitiveType.SPHERE: "models/powerups/health/medium_sphere.md3",
            PrimitiveType.CYLINDER: "models/mapobjects/barrel/barrel.md3",
            PrimitiveType.CAPSULE: "models/mapobjects/capsule/capsule.md3",
        }
        
        model = model_map.get(entity.primitive, "models/mapobjects/barrel/barrel.md3")
        
        return [
            f"// {entity.entity_type.value}: {entity.label}",
            "{",
            '"classname" "misc_model"',
            f'"model" "{model}"',
            f'"origin" "{x:.1f} {y:.1f} {z:.1f}"',
            f'"modelscale" "{max(scale) / 32:.2f}"',
            f'"_color" "{r:.2f} {g:.2f} {b:.2f}"',
            f'"message" "{entity.label}"',
            "}",
        ]
    
    def _importance_to_color(self, importance: float) -> Tuple[float, float, float]:
        """Convert importance to RGB color."""
        if importance >= 0.8:
            return (1.0, 0.2, 0.2)  # Red - high
        elif importance >= 0.5:
            return (1.0, 1.0, 0.2)  # Yellow - medium
        elif importance >= 0.2:
            return (0.2, 1.0, 0.2)  # Green - low
        else:
            return (0.5, 0.5, 0.5)  # Gray - very low
