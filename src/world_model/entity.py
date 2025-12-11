"""
Entity representation for the world model.

Entities are spatial primitives (box, sphere, cylinder) with position,
rotation, scale, and importance tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import time
import uuid
import math


class PrimitiveType(Enum):
    """Geometric primitive types for spatial representation."""
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    PLANE = "plane"


class EntityType(Enum):
    """Entity categories for importance weighting."""
    HUMAN = "human"
    PET = "pet"
    FURNITURE = "furniture"
    APPLIANCE = "appliance"
    OBJECT = "object"
    ZONE = "zone"
    WALL = "wall"
    DOORWAY = "doorway"
    ROBOT = "robot"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Spatial and semantic relationships between entities."""
    ADJACENT_TO = "adjacent_to"
    ON_TOP_OF = "on_top_of"
    INSIDE = "inside"
    PAIRED_WITH = "paired_with"
    FACES = "faces"
    NEAR = "near"


@dataclass
class Relationship:
    """Relationship between two entities."""
    target_id: str
    rel_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """
    Spatial entity in the world model.
    
    Represents objects, people, zones, etc. as geometric primitives
    with importance tracking for memory management.
    """
    
    # Core identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    
    # Geometric representation
    primitive: PrimitiveType = PrimitiveType.BOX
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z in meters
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll in radians
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)     # width, height, depth in meters
    
    # Memory management
    importance: float = 0.5  # 0-1, used for pruning decisions
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    interaction_count: int = 0
    
    # Properties and relationships
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    archived: bool = False
    
    def __post_init__(self):
        """Set default label if not provided."""
        if not self.label:
            self.label = f"{self.entity_type.value}_{self.id[:8]}"
    
    @property
    def age_seconds(self) -> float:
        """Time since entity was created."""
        return time.time() - self.created_at
    
    @property
    def age_days(self) -> float:
        """Age in days."""
        return self.age_seconds / 86400
    
    @property
    def staleness_seconds(self) -> float:
        """Time since last update."""
        return time.time() - self.last_updated
    
    @property
    def staleness_days(self) -> float:
        """Staleness in days."""
        return self.staleness_seconds / 86400
    
    def update(self, **kwargs):
        """Update entity properties and refresh timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = time.time()
    
    def boost_importance(self, amount: float = 0.1):
        """Increase importance (capped at 1.0)."""
        self.importance = min(1.0, self.importance + amount)
        self.last_updated = time.time()
    
    def record_interaction(self):
        """Track that this entity was interacted with."""
        self.interaction_count += 1
        self.boost_importance(0.05)
        self.last_updated = time.time()
    
    def add_relationship(self, target_id: str, rel_type: RelationType, **metadata):
        """Add a relationship to another entity."""
        rel = Relationship(
            target_id=target_id,
            rel_type=rel_type,
            metadata=metadata
        )
        self.relationships.append(rel)
        self.last_updated = time.time()
    
    def remove_relationship(self, target_id: str, rel_type: Optional[RelationType] = None):
        """Remove relationship(s) with target entity."""
        if rel_type:
            self.relationships = [
                r for r in self.relationships
                if not (r.target_id == target_id and r.rel_type == rel_type)
            ]
        else:
            self.relationships = [
                r for r in self.relationships
                if r.target_id != target_id
            ]
        self.last_updated = time.time()
    
    def get_relationships(self, rel_type: Optional[RelationType] = None) -> List[Relationship]:
        """Get all relationships, optionally filtered by type."""
        if rel_type:
            return [r for r in self.relationships if r.rel_type == rel_type]
        return self.relationships
    
    def distance_to(self, other: 'Entity') -> float:
        """Calculate Euclidean distance to another entity."""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        dz = self.position[2] - other.position[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def distance_to_point(self, point: Tuple[float, float, float]) -> float:
        """Calculate distance to a point."""
        dx = self.position[0] - point[0]
        dy = self.position[1] - point[1]
        dz = self.position[2] - point[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_importance(
        self,
        recency_decay: float = 7.0,  # days
        base_weights: Optional[Dict[EntityType, float]] = None
    ) -> float:
        """
        Calculate current importance based on multiple factors.
        
        Args:
            recency_decay: Number of days for importance to decay to ~37%
            base_weights: Entity type importance weights
        
        Returns:
            Calculated importance score (0-1)
        """
        if base_weights is None:
            base_weights = {
                EntityType.HUMAN: 1.0,
                EntityType.PET: 0.9,
                EntityType.ROBOT: 0.8,
                EntityType.FURNITURE: 0.5,
                EntityType.APPLIANCE: 0.5,
                EntityType.OBJECT: 0.3,
                EntityType.ZONE: 0.4,
                EntityType.WALL: 0.6,
                EntityType.DOORWAY: 0.7,
                EntityType.UNKNOWN: 0.2,
            }
        
        # Base importance from entity type
        base = base_weights.get(self.entity_type, 0.3)
        
        # Recency factor (exponential decay)
        recency = math.exp(-self.staleness_days / recency_decay)
        
        # Interaction frequency (logarithmic)
        interaction_weight = math.log(1 + self.interaction_count) / 10.0
        interaction_weight = min(1.0, interaction_weight)
        
        # Combine factors
        calculated = base * recency * (0.7 + 0.3 * interaction_weight)
        
        # Weight with current importance (80% calculated, 20% current)
        return 0.8 * calculated + 0.2 * self.importance
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        return {
            'id': self.id,
            'label': self.label,
            'entity_type': self.entity_type.value,
            'primitive': self.primitive.value,
            'position': list(self.position),
            'rotation': list(self.rotation),
            'scale': list(self.scale),
            'importance': self.importance,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'interaction_count': self.interaction_count,
            'properties': self.properties,
            'relationships': [
                {
                    'target_id': r.target_id,
                    'type': r.rel_type.value,
                    'metadata': r.metadata
                }
                for r in self.relationships
            ],
            'notes': self.notes,
            'tags': self.tags,
            'archived': self.archived
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Deserialize entity from dictionary."""
        # Convert enums
        entity_type = EntityType(data['entity_type'])
        primitive = PrimitiveType(data['primitive'])
        
        # Convert relationships
        relationships = [
            Relationship(
                target_id=r['target_id'],
                rel_type=RelationType(r['type']),
                metadata=r.get('metadata', {})
            )
            for r in data.get('relationships', [])
        ]
        
        return cls(
            id=data['id'],
            label=data['label'],
            entity_type=entity_type,
            primitive=primitive,
            position=tuple(data['position']),
            rotation=tuple(data['rotation']),
            scale=tuple(data['scale']),
            importance=data['importance'],
            created_at=data['created_at'],
            last_updated=data['last_updated'],
            interaction_count=data.get('interaction_count', 0),
            properties=data.get('properties', {}),
            relationships=relationships,
            notes=data.get('notes', ''),
            tags=data.get('tags', []),
            archived=data.get('archived', False)
        )
    
    def __repr__(self) -> str:
        return (
            f"Entity({self.label}, type={self.entity_type.value}, "
            f"pos={self.position}, importance={self.importance:.2f})"
        )
