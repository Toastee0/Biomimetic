"""
Spatial database for world model entities.

Stores entities with spatial indexing for efficient queries.
Uses SQLite for persistence and in-memory spatial structures for fast lookups.
"""

import sqlite3
import json
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import time
import math

from .entity import Entity, EntityType, PrimitiveType, RelationType


class SpatialDB:
    """
    Spatial database for world model entities.
    
    Combines SQLite persistence with in-memory spatial indexing
    for efficient queries.
    """
    
    def __init__(self, db_path: str = "data/world_model/spatial.db"):
        """
        Initialize spatial database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._init_schema()
        self._entity_cache: Dict[str, Entity] = {}
        self._load_entities()
    
    def _init_schema(self):
        """Create database schema if not exists."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    primitive TEXT NOT NULL,
                    pos_x REAL NOT NULL,
                    pos_y REAL NOT NULL,
                    pos_z REAL NOT NULL,
                    rot_pitch REAL NOT NULL,
                    rot_yaw REAL NOT NULL,
                    rot_roll REAL NOT NULL,
                    scale_x REAL NOT NULL,
                    scale_y REAL NOT NULL,
                    scale_z REAL NOT NULL,
                    importance REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    interaction_count INTEGER NOT NULL DEFAULT 0,
                    properties TEXT,
                    relationships TEXT,
                    notes TEXT,
                    tags TEXT,
                    archived INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            # Spatial index on position
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_position
                ON entities(pos_x, pos_y, pos_z)
            """)
            
            # Index on importance for pruning queries
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance
                ON entities(importance DESC, last_updated DESC)
            """)
            
            # Index on type for filtering
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_type
                ON entities(entity_type, archived)
            """)
            
            # Archive table for pruned entities
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_archive (
                    id TEXT PRIMARY KEY,
                    entity_data TEXT NOT NULL,
                    archived_at REAL NOT NULL,
                    reason TEXT
                )
            """)
    
    def _load_entities(self):
        """Load all non-archived entities into memory cache."""
        cursor = self.conn.execute(
            "SELECT * FROM entities WHERE archived = 0"
        )
        
        for row in cursor:
            entity = self._row_to_entity(row)
            self._entity_cache[entity.id] = entity
    
    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert database row to Entity object."""
        properties = json.loads(row['properties']) if row['properties'] else {}
        relationships_data = json.loads(row['relationships']) if row['relationships'] else []
        tags = json.loads(row['tags']) if row['tags'] else []
        
        from .entity import Relationship
        relationships = [
            Relationship(
                target_id=r['target_id'],
                rel_type=RelationType(r['type']),
                metadata=r.get('metadata', {})
            )
            for r in relationships_data
        ]
        
        return Entity(
            id=row['id'],
            label=row['label'],
            entity_type=EntityType(row['entity_type']),
            primitive=PrimitiveType(row['primitive']),
            position=(row['pos_x'], row['pos_y'], row['pos_z']),
            rotation=(row['rot_pitch'], row['rot_yaw'], row['rot_roll']),
            scale=(row['scale_x'], row['scale_y'], row['scale_z']),
            importance=row['importance'],
            created_at=row['created_at'],
            last_updated=row['last_updated'],
            interaction_count=row['interaction_count'],
            properties=properties,
            relationships=relationships,
            notes=row['notes'] or '',
            tags=tags,
            archived=bool(row['archived'])
        )
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add or update entity in database.
        
        Args:
            entity: Entity to add/update
        
        Returns:
            Entity ID
        """
        relationships_json = json.dumps([
            {
                'target_id': r.target_id,
                'type': r.rel_type.value,
                'metadata': r.metadata
            }
            for r in entity.relationships
        ])
        
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO entities (
                    id, label, entity_type, primitive,
                    pos_x, pos_y, pos_z,
                    rot_pitch, rot_yaw, rot_roll,
                    scale_x, scale_y, scale_z,
                    importance, created_at, last_updated, interaction_count,
                    properties, relationships, notes, tags, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id, entity.label, entity.entity_type.value, entity.primitive.value,
                entity.position[0], entity.position[1], entity.position[2],
                entity.rotation[0], entity.rotation[1], entity.rotation[2],
                entity.scale[0], entity.scale[1], entity.scale[2],
                entity.importance, entity.created_at, entity.last_updated, entity.interaction_count,
                json.dumps(entity.properties), relationships_json,
                entity.notes, json.dumps(entity.tags), int(entity.archived)
            ))
        
        self._entity_cache[entity.id] = entity
        return entity.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entity_cache.get(entity_id)
    
    def find_by_label(self, label: str) -> Optional[Entity]:
        """Find entity by label."""
        for entity in self._entity_cache.values():
            if entity.label == label:
                return entity
        return None
    
    def update_entity(self, entity: Entity):
        """Update existing entity."""
        self.add_entity(entity)
    
    def delete_entity(self, entity_id: str):
        """Delete entity from database."""
        with self.conn:
            self.conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        
        if entity_id in self._entity_cache:
            del self._entity_cache[entity_id]
    
    def archive_entity(self, entity_id: str, reason: str = "pruned"):
        """Archive entity (soft delete)."""
        entity = self.get_entity(entity_id)
        if not entity:
            return
        
        # Mark as archived in main table
        entity.archived = True
        self.update_entity(entity)
        
        # Save to archive table
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO entity_archive (id, entity_data, archived_at, reason)
                VALUES (?, ?, ?, ?)
            """, (
                entity_id,
                json.dumps(entity.to_dict()),
                time.time(),
                reason
            ))
        
        # Remove from cache
        if entity_id in self._entity_cache:
            del self._entity_cache[entity_id]
    
    def get_all_entities(
        self,
        entity_type: Optional[EntityType] = None,
        include_archived: bool = False
    ) -> List[Entity]:
        """
        Get all entities, optionally filtered by type.
        
        Args:
            entity_type: Filter by entity type
            include_archived: Include archived entities
        
        Returns:
            List of entities
        """
        entities = list(self._entity_cache.values())
        
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        
        if not include_archived:
            entities = [e for e in entities if not e.archived]
        
        return entities
    
    def query_radius(
        self,
        center: Tuple[float, float, float],
        radius: float,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """
        Find all entities within radius of center point.
        
        Args:
            center: Center point (x, y, z)
            radius: Search radius in meters
            entity_type: Optional type filter
        
        Returns:
            List of entities within radius
        """
        results = []
        radius_sq = radius * radius
        
        for entity in self._entity_cache.values():
            if entity.archived:
                continue
            
            if entity_type and entity.entity_type != entity_type:
                continue
            
            dx = entity.position[0] - center[0]
            dy = entity.position[1] - center[1]
            dz = entity.position[2] - center[2]
            dist_sq = dx*dx + dy*dy + dz*dz
            
            if dist_sq <= radius_sq:
                results.append(entity)
        
        return results
    
    def query_box(
        self,
        min_corner: Tuple[float, float, float],
        max_corner: Tuple[float, float, float],
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """
        Find all entities within axis-aligned bounding box.
        
        Args:
            min_corner: Minimum corner (x, y, z)
            max_corner: Maximum corner (x, y, z)
            entity_type: Optional type filter
        
        Returns:
            List of entities in box
        """
        results = []
        
        for entity in self._entity_cache.values():
            if entity.archived:
                continue
            
            if entity_type and entity.entity_type != entity_type:
                continue
            
            pos = entity.position
            if (min_corner[0] <= pos[0] <= max_corner[0] and
                min_corner[1] <= pos[1] <= max_corner[1] and
                min_corner[2] <= pos[2] <= max_corner[2]):
                results.append(entity)
        
        return results
    
    def nearest_entities(
        self,
        point: Tuple[float, float, float],
        n: int = 5,
        entity_type: Optional[EntityType] = None,
        max_distance: Optional[float] = None
    ) -> List[Tuple[Entity, float]]:
        """
        Find N nearest entities to a point.
        
        Args:
            point: Query point (x, y, z)
            n: Number of results to return
            entity_type: Optional type filter
            max_distance: Maximum distance to consider
        
        Returns:
            List of (entity, distance) tuples, sorted by distance
        """
        distances = []
        
        for entity in self._entity_cache.values():
            if entity.archived:
                continue
            
            if entity_type and entity.entity_type != entity_type:
                continue
            
            distance = entity.distance_to_point(point)
            
            if max_distance and distance > max_distance:
                continue
            
            distances.append((entity, distance))
        
        # Sort by distance and take top N
        distances.sort(key=lambda x: x[1])
        return distances[:n]
    
    def raycast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_distance: float = 10.0
    ) -> List[Tuple[Entity, float]]:
        """
        Cast a ray and find intersecting entities.
        
        Simple implementation: checks distance of entity center to ray.
        
        Args:
            origin: Ray origin (x, y, z)
            direction: Ray direction (x, y, z) - will be normalized
            max_distance: Maximum ray length
        
        Returns:
            List of (entity, distance) tuples along ray
        """
        # Normalize direction
        length = math.sqrt(sum(d*d for d in direction))
        if length == 0:
            return []
        dir_norm = tuple(d / length for d in direction)
        
        hits = []
        
        for entity in self._entity_cache.values():
            if entity.archived:
                continue
            
            # Vector from origin to entity
            to_entity = tuple(
                entity.position[i] - origin[i]
                for i in range(3)
            )
            
            # Project onto ray direction
            projection = sum(to_entity[i] * dir_norm[i] for i in range(3))
            
            if projection < 0 or projection > max_distance:
                continue
            
            # Point on ray closest to entity
            closest_point = tuple(
                origin[i] + projection * dir_norm[i]
                for i in range(3)
            )
            
            # Distance from entity to ray
            dist_to_ray = entity.distance_to_point(closest_point)
            
            # Use entity scale as hit radius (rough approximation)
            hit_radius = max(entity.scale) / 2
            
            if dist_to_ray <= hit_radius:
                hits.append((entity, projection))
        
        # Sort by distance along ray
        hits.sort(key=lambda x: x[1])
        return hits
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total = len(self._entity_cache)
        
        by_type = {}
        for entity in self._entity_cache.values():
            type_name = entity.entity_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM entity_archive")
        archived_count = cursor.fetchone()[0]
        
        return {
            'total_entities': total,
            'by_type': by_type,
            'archived': archived_count,
            'db_path': self.db_path
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()
