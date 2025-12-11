"""
Quake 3 Bridge Module

Connects World Model spatial database to ioquake3 game engine for
real-time 3D visualization and multiplayer editing.
"""

import socket
import json
import time
import threading
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from ..world_model import WorldModel
from ..entity import Entity, EntityType, PrimitiveType


logger = logging.getLogger(__name__)


class Quake3Bridge:
    """
    Bridge between World Model and Quake 3 engine.
    
    Handles:
    - Entity spawning in Quake 3
    - Real-time sync of entity updates
    - Client command processing
    - Map generation
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        server_host: str = "localhost",
        server_port: int = 27960,
        rcon_password: str = "worldmodel"
    ):
        """
        Initialize Quake 3 bridge.
        
        Args:
            world_model: World Model instance
            server_host: Quake 3 server host
            server_port: Quake 3 server port
            rcon_password: RCON password for server control
        """
        self.world_model = world_model
        self.server_host = server_host
        self.server_port = server_port
        self.rcon_password = rcon_password
        
        self.entity_map: Dict[str, int] = {}  # World Model ID → Quake entity num
        self.running = False
        self.sync_thread: Optional[threading.Thread] = None
    
    def send_rcon(self, command: str) -> str:
        """
        Send RCON command to Quake 3 server.
        
        Args:
            command: Console command to execute
        
        Returns:
            Server response
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        
        # Quake 3 RCON packet format
        packet = f"\xff\xff\xff\xffrcon {self.rcon_password} {command}"
        
        try:
            sock.sendto(packet.encode(), (self.server_host, self.server_port))
            response, _ = sock.recvfrom(8192)
            return response.decode('utf-8', errors='ignore')
        except socket.timeout:
            logger.warning(f"RCON timeout: {command}")
            return ""
        finally:
            sock.close()
    
    def spawn_entity(self, entity: Entity) -> bool:
        """
        Spawn entity in Quake 3 world.
        
        Args:
            entity: Entity to spawn
        
        Returns:
            Success status
        """
        # Map entity type to Quake 3 model
        model = self._get_model_for_entity(entity)
        
        # Convert position (World Model uses Y-up, Quake uses Z-up)
        x, y, z = entity.position
        qx, qy, qz = x, z, y  # Swap Y and Z
        
        # Scale
        scale_x, scale_y, scale_z = entity.scale
        
        # Color based on importance
        r, g, b = self._importance_to_color(entity.importance)
        
        # Spawn command
        command = (
            f"spawnentity {model} "
            f"{qx} {qy} {qz} "
            f"{scale_x} {scale_y} {scale_z} "
            f"{r} {g} {b} "
            f"\"{entity.label}\""
        )
        
        response = self.send_rcon(command)
        
        # Parse entity number from response
        # Format: "Entity spawned: 42"
        if "Entity spawned:" in response:
            try:
                entity_num = int(response.split(":")[-1].strip())
                self.entity_map[entity.id] = entity_num
                logger.info(f"Spawned {entity.label} as entity {entity_num}")
                return True
            except ValueError:
                pass
        
        return False
    
    def update_entity(self, entity: Entity) -> bool:
        """
        Update existing entity in Quake 3.
        
        Args:
            entity: Entity to update
        
        Returns:
            Success status
        """
        if entity.id not in self.entity_map:
            return self.spawn_entity(entity)
        
        entity_num = self.entity_map[entity.id]
        
        # Convert position
        x, y, z = entity.position
        qx, qy, qz = x, z, y
        
        # Update position
        self.send_rcon(f"setentitypos {entity_num} {qx} {qy} {qz}")
        
        # Update color based on importance
        r, g, b = self._importance_to_color(entity.importance)
        self.send_rcon(f"setentitycolor {entity_num} {r} {g} {b}")
        
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete entity from Quake 3.
        
        Args:
            entity_id: World Model entity ID
        
        Returns:
            Success status
        """
        if entity_id not in self.entity_map:
            return False
        
        entity_num = self.entity_map[entity_id]
        self.send_rcon(f"removeentity {entity_num}")
        del self.entity_map[entity_id]
        
        return True
    
    def sync_all_entities(self):
        """Sync all entities from World Model to Quake 3."""
        entities = self.world_model.db.get_all_entities()
        
        for entity in entities:
            self.spawn_entity(entity)
        
        logger.info(f"Synced {len(entities)} entities to Quake 3")
    
    def start_sync_loop(self, interval: float = 1.0):
        """
        Start background thread that syncs entities periodically.
        
        Args:
            interval: Sync interval in seconds
        """
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, args=(interval,))
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        logger.info("Started entity sync loop")
    
    def stop_sync_loop(self):
        """Stop background sync loop."""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join()
        
        logger.info("Stopped entity sync loop")
    
    def _sync_loop(self, interval: float):
        """Background sync loop."""
        last_sync = {}
        
        while self.running:
            entities = self.world_model.db.get_all_entities()
            
            for entity in entities:
                # Only update if entity changed
                if (entity.id not in last_sync or
                    last_sync[entity.id] < entity.last_updated):
                    
                    self.update_entity(entity)
                    last_sync[entity.id] = entity.last_updated
            
            time.sleep(interval)
    
    def _get_model_for_entity(self, entity: Entity) -> str:
        """Get Quake 3 model path for entity type."""
        # Map entity types to models
        models = {
            EntityType.HUMAN: "models/players/sarge/default.md3",
            EntityType.ROBOT: "models/custom/robot.md3",
            EntityType.FURNITURE: "models/mapobjects/furniture/desk.md3",
            EntityType.APPLIANCE: "models/mapobjects/appliance.md3",
            EntityType.OBJECT: "models/mapobjects/generic_box.md3",
            EntityType.ZONE: "models/mapobjects/zone_marker.md3",
        }
        
        # Map primitives to simple models
        primitive_models = {
            PrimitiveType.BOX: "models/primitives/box.md3",
            PrimitiveType.SPHERE: "models/primitives/sphere.md3",
            PrimitiveType.CYLINDER: "models/primitives/cylinder.md3",
            PrimitiveType.CAPSULE: "models/primitives/capsule.md3",
        }
        
        # Try entity type first, fallback to primitive
        return models.get(
            entity.entity_type,
            primitive_models.get(entity.primitive, "models/primitives/box.md3")
        )
    
    def _importance_to_color(self, importance: float) -> Tuple[float, float, float]:
        """
        Convert importance to RGB color.
        
        Red = high, Yellow = medium, Green = low, Gray = very low
        """
        if importance >= 0.8:
            # High: Red
            return (1.0, 0.2, 0.2)
        elif importance >= 0.5:
            # Medium: Yellow
            return (1.0, 1.0, 0.2)
        elif importance >= 0.2:
            # Low: Green
            return (0.2, 1.0, 0.2)
        else:
            # Very low: Gray
            return (0.5, 0.5, 0.5)
    
    def generate_map_file(self, output_path: str):
        """
        Generate Quake 3 .map file from World Model.
        
        Args:
            output_path: Path to write .map file
        """
        entities = self.world_model.db.get_all_entities()
        
        lines = []
        lines.append("// World Model Generated Map")
        lines.append("// Generated by BioMimetic AI World Model")
        lines.append("{")
        lines.append('"classname" "worldspawn"')
        lines.append('"message" "AI World Model"')
        lines.append("")
        
        # Generate floor
        lines.extend(self._generate_floor_brush())
        lines.append("}")
        lines.append("")
        
        # Generate entity definitions
        for entity in entities:
            lines.extend(self._generate_entity_definition(entity))
            lines.append("")
        
        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated map file: {output_path}")
    
    def _generate_floor_brush(self) -> List[str]:
        """Generate floor brush definition."""
        return [
            "// Floor",
            "{",
            "brushDef",
            "{",
            "( 0 0 1 -32 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) common/caulk 0 0 0",
            "( 0 0 -1 -32 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) textures/base_floor/conc 0 0 0",
            "( 0 1 0 -512 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) common/caulk 0 0 0",
            "( 0 -1 0 -512 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) common/caulk 0 0 0",
            "( 1 0 0 -512 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) common/caulk 0 0 0",
            "( -1 0 0 -512 ) ( ( 0.03125 0 0 ) ( 0 0.03125 0 ) ) common/caulk 0 0 0",
            "}",
            "}",
        ]
    
    def _generate_entity_definition(self, entity: Entity) -> List[str]:
        """Generate entity definition for map file."""
        model = self._get_model_for_entity(entity)
        
        # Convert position
        x, y, z = entity.position
        qx, qy, qz = x * 32, z * 32, y * 32  # Scale to Quake units (1 unit = 32 game units)
        
        return [
            "{",
            '"classname" "misc_model"',
            f'"model" "{model}"',
            f'"origin" "{qx:.1f} {qy:.1f} {qz:.1f}"',
            f'"modelscale" "{entity.scale[0]:.2f}"',
            f'"angle" "{entity.rotation[2] * 180 / 3.14159:.1f}"',  # Convert radians to degrees
            f'"_color" "{self._importance_to_color(entity.importance)[0]:.2f} {self._importance_to_color(entity.importance)[1]:.2f} {self._importance_to_color(entity.importance)[2]:.2f}"',
            f'"targetname" "{entity.label}"',
            "}",
        ]
