"""
Test script for World Model system.

Creates sample entities and demonstrates spatial queries.
"""

import sys
sys.path.insert(0, '/home/toastee/BioMimeticAi/src')

from world_model.world_model import WorldModel
from world_model.entity import EntityType, PrimitiveType
import time


def main():
    print("=== World Model Test ===\n")
    
    # Initialize world model
    wm = WorldModel(
        db_path="data/world_model/test_spatial.db",
        origin=(0.0, 0.0, 0.0)
    )
    
    print("1. Creating test environment...")
    
    # Add walls
    wm.add_entity(
        label="wall_north",
        entity_type=EntityType.WALL,
        position=(0.0, 1.5, -5.0),
        primitive=PrimitiveType.BOX,
        scale=(10.0, 3.0, 0.2),
        properties={"material": "drywall", "color": "white"}
    )
    
    wm.add_entity(
        label="wall_south",
        entity_type=EntityType.WALL,
        position=(0.0, 1.5, 5.0),
        primitive=PrimitiveType.BOX,
        scale=(10.0, 3.0, 0.2)
    )
    
    # Add furniture
    desk = wm.add_entity(
        label="desk_main",
        entity_type=EntityType.FURNITURE,
        position=(0.0, 0.8, 0.0),
        primitive=PrimitiveType.BOX,
        scale=(1.5, 0.1, 0.8),
        properties={"type": "desk", "has_items": True},
        notes="Main workspace desk"
    )
    
    chair = wm.add_entity(
        label="chair_main",
        entity_type=EntityType.FURNITURE,
        position=(0.0, 0.5, 1.0),
        primitive=PrimitiveType.BOX,
        scale=(0.5, 1.0, 0.5),
        properties={"type": "office_chair"}
    )
    
    # Add appliances
    wm.add_entity(
        label="laptop",
        entity_type=EntityType.OBJECT,
        position=(0.3, 0.9, 0.0),
        primitive=PrimitiveType.BOX,
        scale=(0.35, 0.02, 0.25),
        properties={"type": "electronics", "important": True}
    )
    
    # Add a human
    human = wm.add_entity(
        label="human_partner",
        entity_type=EntityType.HUMAN,
        position=(2.0, 1.0, 0.0),
        primitive=PrimitiveType.CAPSULE,
        scale=(0.4, 1.7, 0.4),
        properties={"name": "Partner", "last_seen": "now"},
        importance=1.0
    )
    
    # Add robot embodiment
    wm.add_entity(
        label="rover_bot",
        entity_type=EntityType.ROBOT,
        position=(-1.0, 0.2, 0.5),
        primitive=PrimitiveType.BOX,
        scale=(0.3, 0.2, 0.4),
        properties={"type": "rover", "status": "idle"},
        importance=0.9
    )
    
    # Add some clutter
    for i in range(5):
        wm.add_entity(
            label=f"desk_item_{i}",
            entity_type=EntityType.OBJECT,
            position=(0.2 * i - 0.4, 0.9, 0.1),
            primitive=PrimitiveType.SPHERE,
            scale=(0.05, 0.05, 0.05),
            importance=0.1
        )
    
    print(f"✓ Created test environment\n")
    
    # Stats
    stats = wm.get_stats()
    print("2. World Model Stats:")
    print(f"   Total entities: {stats['total_entities']}")
    print(f"   By type: {stats['by_type']}")
    print()
    
    # Spatial queries
    print("3. Spatial Queries:")
    
    # What's near the origin?
    print("\n   Query: What's within 2m of origin?")
    nearby = wm.query_nearby(radius=2.0)
    for entity in nearby:
        dist = entity.distance_to_point((0, 0, 0))
        print(f"   - {entity.label}: {dist:.2f}m away, importance={entity.importance:.2f}")
    
    # Find nearest entities
    print("\n   Query: 3 nearest entities to origin:")
    nearest = wm.find_nearest(n=3)
    for entity, distance in nearest:
        print(f"   - {entity.label}: {distance:.2f}m")
    
    # Raycast
    print("\n   Query: Raycast forward (direction +X):")
    hits = wm.raycast(direction=(1.0, 0.0, 0.0), max_distance=5.0)
    for entity, distance in hits:
        print(f"   - Hit {entity.label} at {distance:.2f}m")
    
    # Find specific entity
    print("\n   Query: Where is human_partner?")
    partner = wm.find_entity("human_partner")
    if partner:
        print(f"   - Found at position: {partner.position}")
        print(f"   - Importance: {partner.importance}")
    
    print("\n4. Interaction Tracking:")
    print(f"   Laptop interactions before: {wm.find_entity('laptop').interaction_count}")
    wm.record_interaction(wm.find_entity('laptop').id)
    wm.record_interaction(wm.find_entity('laptop').id)
    wm.record_interaction(wm.find_entity('laptop').id)
    print(f"   Laptop interactions after: {wm.find_entity('laptop').interaction_count}")
    print(f"   Laptop importance boosted to: {wm.find_entity('laptop').importance:.2f}")
    
    print("\n5. Memory Management:")
    print(f"   Current entity count: {stats['total_entities']}")
    
    # Update importance
    print("   Updating importance scores...")
    wm.update_importance(recency_decay=7.0)
    
    # Show importance rankings
    entities = wm.db.get_all_entities()
    entities.sort(key=lambda e: e.importance, reverse=True)
    print("\n   Top 5 by importance:")
    for i, entity in enumerate(entities[:5], 1):
        print(f"   {i}. {entity.label}: {entity.importance:.3f}")
    
    print("\n   Bottom 5 by importance:")
    for i, entity in enumerate(entities[-5:], 1):
        print(f"   {i}. {entity.label}: {entity.importance:.3f}")
    
    # Prune low-importance
    print("\n   Pruning entities with importance < 0.15...")
    pruned = wm.prune_entities(threshold=0.15)
    print(f"   ✓ Pruned {pruned} entities")
    
    new_stats = wm.get_stats()
    print(f"   Remaining entities: {new_stats['total_entities']}")
    print(f"   Archived: {new_stats['archived']}")
    
    print("\n6. Export/Import:")
    export_path = "data/world_model/test_snapshot.json"
    wm.export_snapshot(export_path)
    print(f"   ✓ Exported world snapshot to {export_path}")
    
    print("\n=== Test Complete ===")
    
    wm.close()


if __name__ == "__main__":
    main()
