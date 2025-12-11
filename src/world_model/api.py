"""
World Model API Server

FastAPI server providing HTTP and WebSocket interfaces to the world model.
Enables real-time updates and queries from the Core AI system.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import logging
import time
from datetime import datetime

from .world_model import WorldModel
from .entity import Entity, EntityType, PrimitiveType, RelationType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="World Model API",
    description="Spatial memory system for BioMimetic AI",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global world model instance
world_model: Optional[WorldModel] = None


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


# Pydantic models for API
class Vector3(BaseModel):
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class EntityCreate(BaseModel):
    label: str
    entity_type: str  # EntityType enum value
    position: Vector3
    primitive: str = "box"  # PrimitiveType enum value
    scale: Vector3 = Vector3(x=1.0, y=1.0, z=1.0)
    rotation: Vector3 = Vector3(x=0.0, y=0.0, z=0.0)
    importance: float = 0.5
    properties: Dict[str, Any] = {}
    notes: str = ""
    tags: List[str] = []


class EntityUpdate(BaseModel):
    position: Optional[Vector3] = None
    rotation: Optional[Vector3] = None
    scale: Optional[Vector3] = None
    importance: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class RadiusQuery(BaseModel):
    center: Vector3
    radius: float
    entity_type: Optional[str] = None


class RaycastQuery(BaseModel):
    origin: Vector3
    direction: Vector3
    max_distance: float = 10.0


class EntityResponse(BaseModel):
    id: str
    label: str
    entity_type: str
    position: Vector3
    rotation: Vector3
    scale: Vector3
    importance: float
    created_at: float
    last_updated: float
    interaction_count: int
    properties: Dict[str, Any]
    notes: str
    tags: List[str]


def entity_to_response(entity: Entity) -> EntityResponse:
    """Convert Entity to API response model."""
    return EntityResponse(
        id=entity.id,
        label=entity.label,
        entity_type=entity.entity_type.value,
        position=Vector3(x=entity.position[0], y=entity.position[1], z=entity.position[2]),
        rotation=Vector3(x=entity.rotation[0], y=entity.rotation[1], z=entity.rotation[2]),
        scale=Vector3(x=entity.scale[0], y=entity.scale[1], z=entity.scale[2]),
        importance=entity.importance,
        created_at=entity.created_at,
        last_updated=entity.last_updated,
        interaction_count=entity.interaction_count,
        properties=entity.properties,
        notes=entity.notes,
        tags=entity.tags
    )


@app.on_event("startup")
async def startup_event():
    """Initialize world model on startup."""
    global world_model
    world_model = WorldModel()
    logger.info("World Model API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Close world model on shutdown."""
    if world_model:
        world_model.close()
    logger.info("World Model API shutdown")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "World Model API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get world model statistics."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    return world_model.get_stats()


@app.post("/entities", response_model=EntityResponse)
async def create_entity(entity_data: EntityCreate):
    """Create new entity."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    try:
        entity = world_model.add_entity(
            label=entity_data.label,
            entity_type=EntityType(entity_data.entity_type),
            position=entity_data.position.to_tuple(),
            primitive=PrimitiveType(entity_data.primitive),
            scale=entity_data.scale.to_tuple(),
            rotation=entity_data.rotation.to_tuple(),
            importance=entity_data.importance,
            properties=entity_data.properties,
            notes=entity_data.notes,
            tags=entity_data.tags
        )
        
        # Broadcast update to WebSocket clients
        await manager.broadcast({
            "type": "entity_created",
            "entity": entity.to_dict(),
            "timestamp": time.time()
        })
        
        return entity_to_response(entity)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/entities", response_model=List[EntityResponse])
async def list_entities(
    entity_type: Optional[str] = None,
    limit: int = 100
):
    """List all entities."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    filter_type = EntityType(entity_type) if entity_type else None
    entities = world_model.db.get_all_entities(entity_type=filter_type)
    
    return [entity_to_response(e) for e in entities[:limit]]


@app.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: str):
    """Get entity by ID."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    entity = world_model.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity_to_response(entity)


@app.get("/entities/label/{label}", response_model=EntityResponse)
async def get_entity_by_label(label: str):
    """Get entity by label."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    entity = world_model.find_entity(label)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity_to_response(entity)


@app.patch("/entities/{entity_id}", response_model=EntityResponse)
async def update_entity(entity_id: str, update_data: EntityUpdate):
    """Update entity."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    updates = {}
    if update_data.position:
        updates['position'] = update_data.position.to_tuple()
    if update_data.rotation:
        updates['rotation'] = update_data.rotation.to_tuple()
    if update_data.scale:
        updates['scale'] = update_data.scale.to_tuple()
    if update_data.importance is not None:
        updates['importance'] = update_data.importance
    if update_data.properties is not None:
        updates['properties'] = update_data.properties
    if update_data.notes is not None:
        updates['notes'] = update_data.notes
    
    entity = world_model.update_entity(entity_id, **updates)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Broadcast update
    await manager.broadcast({
        "type": "entity_updated",
        "entity": entity.to_dict(),
        "timestamp": time.time()
    })
    
    return entity_to_response(entity)


@app.delete("/entities/{entity_id}")
async def delete_entity(entity_id: str):
    """Delete entity."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    entity = world_model.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    world_model.db.delete_entity(entity_id)
    
    # Broadcast deletion
    await manager.broadcast({
        "type": "entity_deleted",
        "entity_id": entity_id,
        "timestamp": time.time()
    })
    
    return {"status": "deleted", "entity_id": entity_id}


@app.post("/entities/{entity_id}/interact")
async def interact_with_entity(entity_id: str):
    """Record interaction with entity (boosts importance)."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    world_model.record_interaction(entity_id)
    entity = world_model.get_entity(entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return entity_to_response(entity)


@app.post("/query/radius", response_model=List[EntityResponse])
async def query_radius(query: RadiusQuery):
    """Find entities within radius."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    entity_type = EntityType(query.entity_type) if query.entity_type else None
    entities = world_model.query_nearby(
        center=query.center.to_tuple(),
        radius=query.radius,
        entity_type=entity_type
    )
    
    return [entity_to_response(e) for e in entities]


@app.post("/query/raycast")
async def query_raycast(query: RaycastQuery):
    """Cast ray and find intersecting entities."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    hits = world_model.raycast(
        origin=query.origin.to_tuple(),
        direction=query.direction.to_tuple(),
        max_distance=query.max_distance
    )
    
    return [
        {
            "entity": entity_to_response(entity),
            "distance": distance
        }
        for entity, distance in hits
    ]


@app.post("/consolidate")
async def consolidate_memory():
    """Trigger memory consolidation (prune low-importance entities)."""
    if not world_model:
        raise HTTPException(status_code=500, detail="World model not initialized")
    
    stats = world_model.consolidate_memory()
    
    # Broadcast consolidation event
    await manager.broadcast({
        "type": "memory_consolidated",
        "stats": stats,
        "timestamp": time.time()
    })
    
    return stats


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    
    Clients can send queries and receive real-time entity updates.
    """
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            msg_type = data.get("type")
            
            if msg_type == "query_radius":
                params = data.get("params", {})
                center = tuple(params.get("center", [0, 0, 0]))
                radius = params.get("radius", 5.0)
                
                entities = world_model.query_nearby(center=center, radius=radius)
                
                await websocket.send_json({
                    "type": "query_response",
                    "entities": [e.to_dict() for e in entities],
                    "timestamp": time.time()
                })
            
            elif msg_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
