"""
World Model Manager

Main interface for the spatial memory system. Handles entity management,
importance decay, pruning, and integration with other systems.
"""

import time
import math
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from .entity import Entity, EntityType, PrimitiveType, RelationType
from .spatial_db import SpatialDB


logger = logging.getLogger(__name__)


class WorldModel:
    """
    World Model Manager
    
    Central interface for spatial memory system. Manages entities,
    importance-based retention, and spatial queries.
    """
    
    def __init__(
        self,
        db_path: str = "data/world_model/spatial.db",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        retention_threshold: float = 0.2,
        max_entities: int = 500
    ):
        """
        Initialize world model.
        
        Args:
            db_path: Path to spatial database
            origin: World origin point (AI's primary location)
            retention_threshold: Minimum importance to keep (0-1)
            max_entities: Maximum entities to keep in memory
        """
        self.db = SpatialDB(db_path)
        self.origin = origin
        self.retention_threshold = retention_threshold
        self.max_entities = max_entities
        
        logger.info(f"World Model initialized: {self.db.get_stats()}")
    
    def add_entity(
        self,
        label: str,
        entity_type: EntityType,
        position: Tuple[float, float, float],
        primitive: PrimitiveType = PrimitiveType.BOX,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs
    ) -> Entity:
        """
        Add new entity to world model.
        
        Args:
            label: Entity label/name
            entity_type: Type of entity
            position: Position (x, y, z)
            primitive: Geometric primitive type
            scale: Size (width, height, depth)
            **kwargs: Additional entity properties
        
        Returns:
            Created entity
        """
        entity = Entity(
            label=label,
            entity_type=entity_type,
            position=position,
            primitive=primitive,
            scale=scale,
            **kwargs
        )
        
        self.db.add_entity(entity)
        logger.info(f"Added entity: {entity}")
        
        return entity
    
    def update_entity(
        self,
        entity_id: str,
        **kwargs
    ) -> Optional[Entity]:
        """
        Update existing entity.
        
        Args:
            entity_id: Entity ID
            **kwargs: Fields to update
        
        Returns:
            Updated entity or None if not found
        """
        entity = self.db.get_entity(entity_id)
        if not entity:
            logger.warning(f"Entity not found: {entity_id}")
            return None
        
        entity.update(**kwargs)
        self.db.update_entity(entity)
        
        return entity
    
    def find_entity(self, label: str) -> Optional[Entity]:
        """Find entity by label."""
        return self.db.find_by_label(label)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.db.get_entity(entity_id)
    
    def record_interaction(self, entity_id: str):
        """Record that an entity was interacted with."""
        entity = self.db.get_entity(entity_id)
        if entity:
            entity.record_interaction()
            self.db.update_entity(entity)
    
    def query_nearby(
        self,
        center: Optional[Tuple[float, float, float]] = None,
        radius: float = 5.0,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """
        Find entities near a point.
        
        Args:
            center: Center point (defaults to origin)
            radius: Search radius in meters
            entity_type: Optional type filter
        
        Returns:
            List of nearby entities
        """
        if center is None:
            center = self.origin
        
        return self.db.query_radius(center, radius, entity_type)
    
    def find_nearest(
        self,
        point: Optional[Tuple[float, float, float]] = None,
        n: int = 5,
        entity_type: Optional[EntityType] = None
    ) -> List[Tuple[Entity, float]]:
        """
        Find N nearest entities to a point.
        
        Args:
            point: Query point (defaults to origin)
            n: Number of results
            entity_type: Optional type filter
        
        Returns:
            List of (entity, distance) tuples
        """
        if point is None:
            point = self.origin
        
        return self.db.nearest_entities(point, n, entity_type)
    
    def raycast(
        self,
        origin: Optional[Tuple[float, float, float]] = None,
        direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        max_distance: float = 10.0
    ) -> List[Tuple[Entity, float]]:
        """
        Cast ray and find intersecting entities.
        
        Args:
            origin: Ray origin (defaults to world origin)
            direction: Ray direction
            max_distance: Maximum ray length
        
        Returns:
            List of (entity, distance) tuples
        """
        if origin is None:
            origin = self.origin
        
        return self.db.raycast(origin, direction, max_distance)
    
    def update_importance(self, recency_decay: float = 7.0):
        """
        Recalculate importance for all entities.
        
        Args:
            recency_decay: Days for importance to decay to ~37%
        """
        entities = self.db.get_all_entities()
        
        for entity in entities:
            new_importance = entity.calculate_importance(recency_decay)
            if abs(new_importance - entity.importance) > 0.01:
                entity.importance = new_importance
                self.db.update_entity(entity)
        
        logger.info(f"Updated importance for {len(entities)} entities")
    
    def prune_entities(
        self,
        threshold: Optional[float] = None,
        keep_top_n: Optional[int] = None,
        keep_types: Optional[List[EntityType]] = None
    ) -> int:
        """
        Prune low-importance entities.
        
        Args:
            threshold: Minimum importance to keep (defaults to retention_threshold)
            keep_top_n: Keep top N entities by importance
            keep_types: Entity types to always keep (e.g., HUMAN, ROBOT)
        
        Returns:
            Number of entities pruned
        """
        if threshold is None:
            threshold = self.retention_threshold
        
        if keep_types is None:
            keep_types = [EntityType.HUMAN, EntityType.ROBOT]
        
        entities = self.db.get_all_entities()
        
        # Sort by importance
        entities.sort(key=lambda e: e.importance, reverse=True)
        
        pruned = 0
        
        for i, entity in enumerate(entities):
            # Always keep certain types
            if entity.entity_type in keep_types:
                continue
            
            # Keep top N
            if keep_top_n and i < keep_top_n:
                continue
            
            # Prune below threshold
            if entity.importance < threshold:
                self.db.archive_entity(entity.id, reason="low_importance")
                pruned += 1
        
        logger.info(f"Pruned {pruned} entities")
        return pruned
    
    def consolidate_memory(self):
        """
        Daily memory consolidation routine.
        
        Like axiom F7 - consolidate memories during idle time.
        Updates importance and prunes low-priority entities.
        """
        logger.info("Starting memory consolidation...")
        
        # Update all importance scores
        self.update_importance()
        
        # Enforce max entities limit
        entities = self.db.get_all_entities()
        if len(entities) > self.max_entities:
            # Prune to 90% of max
            target = int(self.max_entities * 0.9)
            self.prune_entities(keep_top_n=target)
        
        # Prune below threshold
        pruned = self.prune_entities()
        
        stats = self.db.get_stats()
        logger.info(f"Consolidation complete: {stats}")
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return self.db.get_stats()
    
    def export_snapshot(self, filepath: str):
        """Export current world state to JSON."""
        import json
        
        entities = self.db.get_all_entities()
        data = {
            'origin': self.origin,
            'timestamp': time.time(),
            'entity_count': len(entities),
            'entities': [e.to_dict() for e in entities]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported world snapshot to {filepath}")
    
    def import_snapshot(self, filepath: str):
        """Import world state from JSON."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for entity_data in data['entities']:
            entity = Entity.from_dict(entity_data)
            self.db.add_entity(entity)
        
        logger.info(f"Imported {len(data['entities'])} entities from {filepath}")
    
    def close(self):
        """Close world model and database."""
        self.db.close()
