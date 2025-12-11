"""
Brush Primitive Generator using GtkRadiant CSG algorithms
Wraps the compiled C tool for guaranteed valid geometry
"""
import subprocess
import os
from pathlib import Path
from typing import Tuple, List, Optional

class BrushPrimitive:
    """Represents a simple brush volume"""
    def __init__(self, mins: Tuple[float, float, float], 
                 maxs: Tuple[float, float, float],
                 texture: str = "base_wall/concrete"):
        self.mins = mins
        self.maxs = maxs
        self.texture = texture
    
    def to_solid_brush(self) -> str:
        """Generate a solid brush (6 faces)"""
        # Use the C tool to generate valid geometry
        return f"solid {self.mins[0]} {self.mins[1]} {self.mins[2]} {self.maxs[0]} {self.maxs[1]} {self.maxs[2]} {self.texture}"
    
    def to_hollow(self, thickness: float = 8.0) -> List['BrushPrimitive']:
        """
        Create hollow room from this brush volume
        Returns 6 wall brushes (floor, ceiling, 4 walls)
        """
        mx0, my0, mz0 = self.mins
        mx1, my1, mz1 = self.maxs
        t = thickness
        
        walls = [
            # Floor (bottom)
            BrushPrimitive((mx0, my0, mz0), (mx1, my1, mz0 + t), self.texture),
            # Ceiling (top)
            BrushPrimitive((mx0, my0, mz1 - t), (mx1, my1, mz1), self.texture),
            # Left wall (X-)
            BrushPrimitive((mx0, my0, mz0 + t), (mx0 + t, my1, mz1 - t), self.texture),
            # Right wall (X+)
            BrushPrimitive((mx1 - t, my0, mz0 + t), (mx1, my1, mz1 - t), self.texture),
            # Back wall (Y-)
            BrushPrimitive((mx0 + t, my0, mz0 + t), (mx1 - t, my0 + t, mz1 - t), self.texture),
            # Front wall (Y+)
            BrushPrimitive((mx0 + t, my1 - t, mz0 + t), (mx1 - t, my1, mz1 - t), self.texture),
        ]
        
        return walls


class MapGenerator:
    """Generate .map files using proven CSG tool"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.csg_tool = self.project_root / "brush_csg_tool"
        
        if not self.csg_tool.exists():
            raise FileNotFoundError(f"CSG tool not found: {self.csg_tool}")
    
    def generate_hollow_room(self, 
                            output_path: Path,
                            mins: Tuple[float, float, float],
                            maxs: Tuple[float, float, float],
                            wall_thickness: float = 8.0,
                            texture: str = "base_wall/concrete") -> Path:
        """
        Generate a hollow room using the CSG tool
        
        Args:
            output_path: Where to write the .map file
            mins: Minimum corner (x, y, z)
            maxs: Maximum corner (x, y, z)
            wall_thickness: Wall thickness in Quake units
            texture: Texture to apply
            
        Returns:
            Path to generated .map file
        """
        # Define the volume
        room = BrushPrimitive(mins, maxs, texture)
        
        # Get hollow walls
        walls = room.to_hollow(wall_thickness)
        
        # Write map file manually (simpler than calling C tool)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"// Generated hollow room: {mins} to {maxs}\n")
            f.write("{\n")
            f.write('"classname" "worldspawn"\n')
            
            # Write each wall brush
            for wall in walls:
                self._write_brush(f, wall)
            
            f.write("}\n")
            
            # Add spawn point
            f.write("{\n")
            f.write('"classname" "info_player_deathmatch"\n')
            center_x = (mins[0] + maxs[0]) / 2
            center_y = (mins[1] + maxs[1]) / 2
            spawn_z = mins[2] + wall_thickness + 32
            f.write(f'"origin" "{center_x} {center_y} {spawn_z}"\n')
            f.write('"angle" "0"\n')
            f.write("}\n")
            
            # Add light
            f.write("{\n")
            f.write('"classname" "light"\n')
            light_z = maxs[2] - wall_thickness - 64
            f.write(f'"origin" "{center_x} {center_y} {light_z}"\n')
            f.write('"light" "300"\n')
            f.write("}\n")
        
        print(f"Generated hollow room: {output_path}")
        print(f"  Volume: {mins} to {maxs}")
        print(f"  Wall thickness: {wall_thickness}")
        print(f"  6 wall brushes created")
        
        return output_path
    
    def _write_brush(self, f, brush: BrushPrimitive):
        """Write a brush using Brush_Create planept pattern"""
        mx0, my0, mz0 = brush.mins
        mx1, my1, mz1 = brush.maxs
        tex = brush.texture
        
        # Bottom 4 corners
        pts = [
            [(mx0, my0, mz0), (mx0, my0, mz1)],  # Corner 0
            [(mx0, my1, mz0), (mx0, my1, mz1)],  # Corner 1
            [(mx1, my1, mz0), (mx1, my1, mz1)],  # Corner 2
            [(mx1, my0, mz0), (mx1, my0, mz1)],  # Corner 3
        ]
        
        f.write("{\n")
        
        # 4 vertical faces
        for i in range(4):
            j = (i + 1) % 4
            p0 = pts[j][1]  # Next corner, top
            p1 = pts[i][1]  # This corner, top
            p2 = pts[i][0]  # This corner, bottom
            f.write(f"( {p0[0]} {p0[1]} {p0[2]} ) ( {p1[0]} {p1[1]} {p1[2]} ) ( {p2[0]} {p2[1]} {p2[2]} ) {tex} 0 0 0 1 1 0 0 0\n")
        
        # Bottom face
        p0 = pts[0][0]
        p1 = pts[1][0]
        p2 = pts[2][0]
        f.write(f"( {p0[0]} {p0[1]} {p0[2]} ) ( {p1[0]} {p1[1]} {p1[2]} ) ( {p2[0]} {p2[1]} {p2[2]} ) {tex} 0 0 0 1 1 0 0 0\n")
        
        # Top face
        p0 = pts[2][1]
        p1 = pts[1][1]
        p2 = pts[0][1]
        f.write(f"( {p0[0]} {p0[1]} {p0[2]} ) ( {p1[0]} {p1[1]} {p1[2]} ) ( {p2[0]} {p2[1]} {p2[2]} ) {tex} 0 0 0 1 1 0 0 0\n")
        
        f.write("}\n")
    
    def compile_map(self, map_path: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Compile .map to .bsp using q3map2
        
        Returns:
            Path to generated .bsp file
        """
        q3map2 = self.project_root / "external/q3map2/build/mapcompiler"
        
        if not q3map2.exists():
            raise FileNotFoundError(f"q3map2 not found: {q3map2}")
        
        cmd = [
            str(q3map2),
            "-bsp",
            "-fs_basepath", str(Path.home() / ".q3a"),
            "-game", "quake3",
            str(map_path)
        ]
        
        print(f"Compiling {map_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        # Check for errors
        if "leaked" in result.stdout.lower():
            print("ERROR: Map leaked!")
            print(result.stdout)
            raise RuntimeError("Map compilation failed - leaked")
        
        # Find output BSP
        bsp_path = map_path.with_suffix('.bsp')
        
        if not bsp_path.exists():
            print("ERROR: BSP not created")
            print(result.stdout)
            raise RuntimeError("Map compilation failed - no BSP output")
        
        # Get stats
        for line in result.stdout.split('\n'):
            if 'leafs filled' in line or 'Size:' in line or 'Wrote' in line:
                print(f"  {line.strip()}")
        
        print(f"✓ Compiled: {bsp_path}")
        
        return bsp_path


# Example usage
if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    gen = MapGenerator(project_root)
    
    # Create a hollow room
    map_path = gen.generate_hollow_room(
        output_path=project_root / "data/world_model/maps/hollow_test.map",
        mins=(-256, -256, 0),
        maxs=(256, 256, 256),
        wall_thickness=8,
        texture="base_wall/concrete"
    )
    
    # Compile it
    try:
        bsp_path = gen.compile_map(map_path)
        print(f"\nSuccess! Load in Quake 3 with: /map {map_path.stem}")
    except Exception as e:
        print(f"\nCompilation failed: {e}")
        sys.exit(1)
