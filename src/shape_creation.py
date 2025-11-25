"""Functions to create 3D volumes with spherical shapes for testing."""

# -----------------------------
import numpy as np


# -----------------------------
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


def create_n_spheres_example(centres, radii, intensities, volume_size=10):
    """Create a 3D volume containing multiple spheres.

    Generates a cubic 3D array and adds multiple spheres with specified
    parameters. When spheres overlap, the maximum intensity value is retained
    at each voxel.

    Parameters:
        centres (list of tuples): List of centre coordinates [(cz, cy, cx), ...]
        radii (list of float): List of sphere radii in voxels
        intensities (list of int): List of greyscale intensity values for each sphere
        volume_size (int): Size of the cubic volume (default: 10)

    Returns:
        np.ndarray: 3D uint8 array containing all spheres

    Raises:
        ValueError: If centres, radii, and intensities lists have different lengths
    """
    # Validate input list lengths match
    if not (len(centres) == len(radii) == len(intensities)):
        raise ValueError(
            f"Input lists must have equal length: centres={len(centres)}, "
            f"radii={len(radii)}, intensities={len(intensities)}"
        )

    # Initialise empty volume
    image3d = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint8)

    # Add each sphere to the volume, taking maximum intensity at overlaps
    for centre, radius, intensity in zip(centres, radii, intensities):
        sphere = create_sphere(
            volume_shape=image3d.shape,
            centre=centre,
            radius=radius,
            intensity=intensity,
        )
        image3d = np.maximum(image3d, sphere)

    return image3d


def create_two_spheres_example(centre1, centre2, radius1, radius2):
    """Create a 3D volume containing two spheres.

    Generates a 10x10x10 array containing two spheres with fixed intensities
    of 200 and 180 for the first and second sphere respectively.

    Parameters:
        centre1 (tuple): Centre coordinates (cz, cy, cx) of first sphere
        centre2 (tuple): Centre coordinates (cz, cy, cx) of second sphere
        radius1 (float): Radius of first sphere in voxels
        radius2 (float): Radius of second sphere in voxels

    Returns:
        np.ndarray: 10x10x10 3D uint8 array containing two spheres
    """
    return create_n_spheres_example(
        centres=[centre1, centre2],
        radii=[radius1, radius2],
        intensities=[200, 180],
    )


def create_three_spheres_example(centre1, centre2, centre3, radius1, radius2, radius3):
    """Create a 3D volume containing three spheres.

    Generates a 10x10x10 array containing three spheres with fixed intensities
    of 200, 180, and 160 for the first, second, and third sphere respectively.

    Parameters:
        centre1 (tuple): Centre coordinates (cz, cy, cx) of first sphere
        centre2 (tuple): Centre coordinates (cz, cy, cx) of second sphere
        centre3 (tuple): Centre coordinates (cz, cy, cx) of third sphere
        radius1 (float): Radius of first sphere in voxels
        radius2 (float): Radius of second sphere in voxels
        radius3 (float): Radius of third sphere in voxels

    Returns:
        np.ndarray: 10x10x10 3D uint8 array containing three spheres
    """
    return create_n_spheres_example(
        centres=[centre1, centre2, centre3],
        radii=[radius1, radius2, radius3],
        intensities=[200, 180, 160],
    )
