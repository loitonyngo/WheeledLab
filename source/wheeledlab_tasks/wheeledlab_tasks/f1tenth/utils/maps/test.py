import os
import numpy as np
import yaml
from PIL import Image
# from pxr import Usd, UsdGeom, UsdPhysics, Gf

class TraversabilityHashmapUtil:
    def set_traversability_hashmap(self, hashmap, map_size, spacing):
        """Mock function to store the hashmap (replace with your actual implementation)."""
        self.hashmap = hashmap
        self.map_size = map_size
        self.spacing = spacing

def create_maps_from_png(png_path, yaml_path, output_usd_path):
    """
    Create a USD file and traversability hashmap from a PNG + YAML pair.
    
    Args:
        png_path: Path to the PNG image (black=obstacle, white=drivable).
        yaml_path: Path to the YAML file with metadata (resolution, origin).
        output_usd_path: Output USD file path (e.g., "/path/to/track.usd").
    """
    # Load YAML metadata
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    resolution = yaml_data['resolution']  # meters/pixel
    origin = yaml_data['origin']         # [x, y, z] offset in meters
    negate = yaml_data.get('negate', 0)  # Invert colors if needed

    # Load PNG and convert to binary hashmap
    image = Image.open(png_path).convert('L')  # Grayscale
    image_array = np.array(image)
    
    # Apply negate (if specified in YAML)
    if negate:
        image_array = 255 - image_array
    
    # Threshold to binary (white=drivable, black=obstacle)
    occupied_thresh = yaml_data.get('occupied_thresh', 0.65)
    free_thresh = yaml_data.get('free_thresh', 0.196)
    hashmap = (image_array / 255.0 > free_thresh).astype(bool)

    # Get map dimensions in pixels and meters
    map_size_pixels = image_array.shape  # (height, width)
    map_size_meters = (
        map_size_pixels[0] * resolution,
        map_size_pixels[1] * resolution
    )

    # Generate USD file (adapted from your original function)
    stage = Usd.Stage.CreateNew(output_usd_path)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())
    plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')

    # Create vertices (scaled by resolution and shifted by origin)
    xs = np.linspace(
        origin[0],
        origin[0] + map_size_meters[1],
        map_size_pixels[1]
    )
    ys = np.linspace(
        origin[1],
        origin[1] + map_size_meters[0],
        map_size_pixels[0]
    )
    xx, yy = np.meshgrid(xs, ys)
    vertices = [(x, y, 0) for x, y in zip(xx.ravel(), yy.ravel())]

    # Create faces (same as your original code)
    faces = []
    face_counts = []
    for row in range(map_size_pixels[0] - 1):
        for col in range(map_size_pixels[1] - 1):
            v0 = row * map_size_pixels[1] + col
            v1 = v0 + 1
            v2 = v0 + map_size_pixels[1]
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]

    # Assign colors (white=drivable, black=obstacle)
    colors = [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]  # Black, White
    face_colors = []
    for row in range(map_size_pixels[0] - 1):
        for col in range(map_size_pixels[1] - 1):
            face_colors.append(colors[int(hashmap[row, col])])
    
    # Double colors for triangles
    face_colors_triangle = [c for color_pair in zip(face_colors, face_colors) for c in color_pair]

    # Set mesh attributes
    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors_triangle)

    # Add collision (optional)
    UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    stage.Save()

    # Return hashmap and metadata
    envs_boundaries = []  # Optional: Define if you need sub-regions
    spacing = (resolution, resolution)  # Equal spacing in x/y
    TraversabilityHashmapUtil().set_traversability_hashmap(
        hashmap.tolist(), map_size_pixels, spacing
    )
    return hashmap, envs_boundaries


if __name__ == "__main__":
    path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/f1tenth/utils/maps/f'
    create_maps_from_png('test.usd', 100, 100, 0.3, 0.3, 0.3)
