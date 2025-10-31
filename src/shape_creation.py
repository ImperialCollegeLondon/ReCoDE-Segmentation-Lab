#!/usr/bin/env python3
"""Functions to create example array.

Requirements:
   - Python 3.11
   - Required libraries:
       * numpy

"""

# -----------------------------
import numpy as np

def create_sphere(volume_shape, centre, radius, intensity):
    """Create a sphere in a 3D volume.

    Generates a solid sphere by calculating Euclidean distance from centre
    and filling all voxels within the specified radius.

    Parameters:
        volume_shape (tuple): Shape of the 3D volume (z, y, x)
        centre (tuple): Centre coordinates (cz, cy, cx) of the sphere
        radius (float): Radius of the sphere in voxels
        intensity (int): Greyscale intensity value for the sphere

    Returns:
        np.ndarray: 3D boolean array where True indicates sphere voxels
    """
    # Create coordinate grids for the entire volume
    z, y, x = np.ogrid[: volume_shape[0], : volume_shape[1], : volume_shape[2]]

    # Calculate Euclidean distance from centre for all voxels
    cz, cy, cx = centre
    distance_from_centre = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)

    # Create mask: voxels within radius belong to sphere
    sphere_mask = distance_from_centre <= radius

    return (sphere_mask * intensity).astype(np.uint8)

def create_two_spheres_example():
    volume_size = 10
    image3d = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint8)

    # Add large sphere (intensity 200)
    large_sphere = create_sphere(
        volume_shape=image3d.shape,
        centre=(6,6,6),  # Centred in volume
        radius=2,
        intensity=200,
    )
    image3d = np.maximum(image3d, large_sphere)

    # Add small sphere (intensity 180)
    small_sphere = create_sphere(
        volume_shape=image3d.shape,
        centre=(2, 2, 2),  # Offset position
        radius=1,
        intensity=180,
    )
    image3d = np.maximum(image3d, small_sphere)
    return image3d
