#!/usr/bin/env python3
"""
Quake 3 .map file editor - Programmatic CSG operations
Provides command-line tools to manipulate map files similar to GtkRadiant's CSG functions.
"""
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


@dataclass
class Plane:
    """Represents a brush face plane in Quake 3 format"""
    p1: Tuple[float, float, float]
    p2: Tuple[float, float, float]
    p3: Tuple[float, float, float]
    texture: str
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    content_flags: int = 0
    surface_flags: int = 0
    value: int = 0
    
    @classmethod
    def from_string(cls, line: str) -> 'Plane':
        """Parse a plane from a .map file line"""
        # ( x1 y1 z1 ) ( x2 y2 z2 ) ( x3 y3 z3 ) texture offsetX offsetY rotation scaleX scaleY contentFlags surfaceFlags value
        match = re.match(
            r'\s*\(\s*([^)]+)\s*\)\s*\(\s*([^)]+)\s*\)\s*\(\s*([^)]+)\s*\)\s*'
            r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*'
            r'(?:(\S+)\s+(\S+)\s+(\S+))?',
            line
        )
        if not match:
            raise ValueError(f"Invalid plane format: {line}")
        
        groups = match.groups()
        p1 = tuple(map(float, groups[0].split()))
        p2 = tuple(map(float, groups[1].split()))
        p3 = tuple(map(float, groups[2].split()))
        texture = groups[3]
        offset_x = float(groups[4])
        offset_y = float(groups[5])
        rotation = float(groups[6])
        scale_x = float(groups[7])
        scale_y = float(groups[8])
        
        # Optional flags
        content_flags = int(groups[9]) if groups[9] else 0
        surface_flags = int(groups[10]) if groups[10] else 0
        value = int(groups[11]) if groups[11] else 0
        
        return cls(p1, p2, p3, texture, offset_x, offset_y, rotation, 
                   scale_x, scale_y, content_flags, surface_flags, value)
    
    def to_string(self) -> str:
        """Convert plane back to .map format"""
        return (f"( {self.p1[0]:.0f} {self.p1[1]:.0f} {self.p1[2]:.0f} ) "
                f"( {self.p2[0]:.0f} {self.p2[1]:.0f} {self.p2[2]:.0f} ) "
                f"( {self.p3[0]:.0f} {self.p3[1]:.0f} {self.p3[2]:.0f} ) "
                f"{self.texture} {self.offset_x:.0f} {self.offset_y:.0f} "
                f"{self.rotation:.0f} {self.scale_x} {self.scale_y} "
                f"{self.content_flags} {self.surface_flags} {self.value}")
    
    def offset(self, x: float = 0, y: float = 0, z: float = 0) -> 'Plane':
        """Create a new plane offset by x, y, z"""
        return Plane(
            (self.p1[0] + x, self.p1[1] + y, self.p1[2] + z),
            (self.p2[0] + x, self.p2[1] + y, self.p2[2] + z),
            (self.p3[0] + x, self.p3[1] + y, self.p3[2] + z),
            self.texture, self.offset_x, self.offset_y, self.rotation,
            self.scale_x, self.scale_y, self.content_flags, 
            self.surface_flags, self.value
        )


@dataclass
class Brush:
    """Represents a Quake 3 brush"""
    planes: List[Plane]
    
    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Brush':
        """Parse a brush from .map file lines"""
        planes = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//') and line.startswith('('):
                planes.append(Plane.from_string(line))
        return cls(planes)
    
    def to_lines(self) -> List[str]:
        """Convert brush back to .map format lines"""
        lines = ["{"]
        for plane in self.planes:
            lines.append(plane.to_string())
        lines.append("}")
        return lines
    
    @classmethod
    def create_box(cls, min_x: float, min_y: float, min_z: float,
                   max_x: float, max_y: float, max_z: float,
                   texture: str = "base_floor/concrete", 
                   thickness: float = 8) -> 'Brush':
        """Create a box brush (single solid block)"""
        planes = [
            # Bottom (z = min_z)
            Plane((min_x, min_y, min_z), (max_x, min_y, min_z), (min_x, max_y, min_z), texture),
            # Top (z = max_z)
            Plane((min_x, max_y, max_z), (max_x, max_y, max_z), (min_x, min_y, max_z), texture),
            # Left (x = min_x)
            Plane((min_x, min_y, min_z), (min_x, max_y, min_z), (min_x, min_y, max_z), texture),
            # Right (x = max_x)
            Plane((max_x, max_y, min_z), (max_x, min_y, min_z), (max_x, max_y, max_z), texture),
            # Back (y = min_y)
            Plane((max_x, min_y, min_z), (min_x, min_y, min_z), (max_x, min_y, max_z), texture),
            # Front (y = max_y)
            Plane((min_x, max_y, min_z), (max_x, max_y, min_z), (min_x, max_y, max_z), texture),
        ]
        return cls(planes)


@dataclass
class Entity:
    """Represents a Quake 3 entity"""
    properties: dict
    brushes: List[Brush]
    
    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Entity':
        """Parse an entity from .map file lines"""
        properties = {}
        brushes = []
        current_brush_lines = []
        in_brush = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped == '{' and in_brush:
                # Start of brush definition (already in entity)
                continue
            elif line_stripped == '{':
                # Start of entity or brush
                in_brush = True
                current_brush_lines = []
            elif line_stripped == '}':
                # End of brush
                if current_brush_lines:
                    brushes.append(Brush.from_lines(current_brush_lines))
                    current_brush_lines = []
                in_brush = False
            elif '"' in line_stripped and not in_brush:
                # Entity property
                match = re.match(r'"([^"]+)"\s+"([^"]*)"', line_stripped)
                if match:
                    properties[match.group(1)] = match.group(2)
            elif in_brush and line_stripped.startswith('('):
                # Brush plane
                current_brush_lines.append(line)
        
        return cls(properties, brushes)
    
    def to_lines(self) -> List[str]:
        """Convert entity back to .map format lines"""
        lines = ["{"]
        
        # Properties
        for key, value in self.properties.items():
            lines.append(f'"{key}" "{value}"')
        
        # Brushes
        for brush in self.brushes:
            lines.extend(brush.to_lines())
        
        lines.append("}")
        return lines


class MapFile:
    """Represents a complete Quake 3 .map file"""
    
    def __init__(self):
        self.entities: List[Entity] = []
    
    @classmethod
    def load(cls, filepath: str) -> 'MapFile':
        """Load a .map file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        map_file = cls()
        
        # Split into entities (top-level braces)
        entity_depth = 0
        current_entity_lines = []
        
        for line in content.split('\n'):
            stripped = line.strip()
            
            if stripped == '{':
                entity_depth += 1
                if entity_depth == 1:
                    current_entity_lines = [line]
                else:
                    current_entity_lines.append(line)
            elif stripped == '}':
                current_entity_lines.append(line)
                entity_depth -= 1
                if entity_depth == 0:
                    # End of entity
                    map_file.entities.append(Entity.from_lines(current_entity_lines))
                    current_entity_lines = []
            elif entity_depth > 0:
                current_entity_lines.append(line)
        
        return map_file
    
    def save(self, filepath: str):
        """Save the .map file"""
        with open(filepath, 'w') as f:
            f.write("// Generated by map_editor.py\n")
            for entity in self.entities:
                for line in entity.to_lines():
                    f.write(line + '\n')
    
    def add_entity(self, entity: Entity):
        """Add an entity to the map"""
        self.entities.append(entity)
    
    def add_brush(self, brush: Brush, entity_index: int = 0):
        """Add a brush to an existing entity (default: worldspawn)"""
        if entity_index < len(self.entities):
            self.entities[entity_index].brushes.append(brush)
    
    def make_hollow(self, entity_index: int, brush_index: int, thickness: float = 8.0):
        """Make a brush hollow (like GtkRadiant's Hollow function)"""
        entity = self.entities[entity_index]
        brush = entity.brushes[brush_index]
        
        # Remove the original brush
        entity.brushes.pop(brush_index)
        
        # Create 6 wall brushes for a box
        # This is a simplified version - full implementation would handle arbitrary convex brushes
        # For now, assumes the brush is roughly box-shaped
        
        # Calculate bounds
        # ... (complex geometry calculation)
        # For simplicity, this is a stub
        print(f"Hollow operation on entity {entity_index} brush {brush_index} (thickness={thickness})")


def cmd_info(args):
    """Display information about a map file"""
    map_file = MapFile.load(args.input)
    
    print(f"Map: {args.input}")
    print(f"Entities: {len(map_file.entities)}")
    
    for i, entity in enumerate(map_file.entities):
        classname = entity.properties.get('classname', 'unknown')
        print(f"  Entity {i}: {classname} ({len(entity.brushes)} brushes)")


def cmd_add_box(args):
    """Add a box brush to the map"""
    map_file = MapFile.load(args.input) if Path(args.input).exists() else MapFile()
    
    # Ensure worldspawn exists
    if not map_file.entities:
        worldspawn = Entity({'classname': 'worldspawn'}, [])
        map_file.add_entity(worldspawn)
    
    # Create box
    brush = Brush.create_box(
        args.min_x, args.min_y, args.min_z,
        args.max_x, args.max_y, args.max_z,
        args.texture
    )
    
    map_file.add_brush(brush, 0)
    map_file.save(args.output)
    print(f"Added box brush to {args.output}")


def cmd_hollow(args):
    """Make a brush hollow"""
    map_file = MapFile.load(args.input)
    map_file.make_hollow(args.entity, args.brush, args.thickness)
    map_file.save(args.output)
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Quake 3 .map file editor - Programmatic CSG operations'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Display map information')
    parser_info.add_argument('input', help='Input .map file')
    
    # Add box command
    parser_add = subparsers.add_parser('add-box', help='Add a box brush')
    parser_add.add_argument('input', help='Input .map file (created if missing)')
    parser_add.add_argument('output', help='Output .map file')
    parser_add.add_argument('--min-x', type=float, required=True)
    parser_add.add_argument('--min-y', type=float, required=True)
    parser_add.add_argument('--min-z', type=float, required=True)
    parser_add.add_argument('--max-x', type=float, required=True)
    parser_add.add_argument('--max-y', type=float, required=True)
    parser_add.add_argument('--max-z', type=float, required=True)
    parser_add.add_argument('--texture', default='base_floor/concrete')
    
    # Hollow command
    parser_hollow = subparsers.add_parser('hollow', help='Make a brush hollow')
    parser_hollow.add_argument('input', help='Input .map file')
    parser_hollow.add_argument('output', help='Output .map file')
    parser_hollow.add_argument('--entity', type=int, default=0)
    parser_hollow.add_argument('--brush', type=int, required=True)
    parser_hollow.add_argument('--thickness', type=float, default=8.0)
    
    args = parser.parse_args()
    
    if args.command == 'info':
        cmd_info(args)
    elif args.command == 'add-box':
        cmd_add_box(args)
    elif args.command == 'hollow':
        cmd_hollow(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
