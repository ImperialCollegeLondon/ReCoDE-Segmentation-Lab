#!/usr/bin/env python3
"""Functions for visualisation of the 3D dataset.

Script Name:    visualize_image.py
Author:         David Buchner, Imperial College London
Created:        16/10/2025
Last Modified:  16/10/2025
Description:
Functions to visualize images in 2D and 3D, including
interactive display and metadata extraction.

Requirements:
- Python 3.x
- Required libraries:
* numpy
* matplotlib
* plotly (for 3D visualization)

Notes:
Adapted from previous dataset loading scripts to provide
versatile image visualization tools.

"""

# -----------------------------
# Standard library
import matplotlib

# Set backend before importing pyplot
matplotlib.use("TkAgg")  # or 'Qt5Agg'

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
    
# -----------------------------


def plot_3d_orthogonal_planes(volume, cmap="gray", alpha=1.0, snapshot_view=None):
    """Plot orthogonal planes.

    Plots XY, YZ, XZ planes of a 3D volume manually, splitting each into four quadrants
    with explicit coordinates for testing.
    """
    z, y, x = volume.shape
    cx, cy, cz = x // 2, y // 2, z // 2

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap_func = plt.cm.get_cmap(cmap)

    # # ----------------------
    # # YZ Plane at X = cx
    # # ----------------------
    quadrants_yz = [
        (slice(0, cy), slice(0, cz)),  # Q1
        (slice(cy, y), slice(0, cz)),  # Q2
        (slice(0, cy), slice(cz, z)),  # Q3
        (slice(cy, y), slice(cz, z)),  # Q4
    ]
    for ys, zs in quadrants_yz:
        Y, Z = np.meshgrid(
            np.arange(ys.start, ys.stop), np.arange(zs.start, zs.stop), indexing="ij"
        )
        X_plane = np.ones_like(Y) * cx
        facecolors = cmap_func(volume[zs, ys, cx] / volume.max())
        ax.plot_surface(
            X_plane, Y.T, Z.T, facecolors=facecolors, alpha=alpha, shade=False
        )

    # # ----------------------
    # # # XZ Plane at Y = cy
    # # # ----------------------
    quadrants_xz = [
        (slice(0, cx), slice(0, cz)),  # Q1
        (slice(cx, x), slice(0, cz)),  # Q2
        (slice(0, cx), slice(cz, z)),  # Q3
        (slice(cx, x), slice(cz, z)),  # Q4
    ]
    for xs, zs in quadrants_xz:
        X, Z = np.meshgrid(
            np.arange(xs.start, xs.stop), np.arange(zs.start, zs.stop), indexing="ij"
        )
        Y_plane = np.ones_like(X) * cy
        facecolors = cmap_func(volume[zs, cy, xs] / volume.max())
        ax.plot_surface(
            X.T, Y_plane, Z.T, facecolors=facecolors, alpha=alpha, shade=False
        )

    # ----------------------
    # Axes and view
    # ----------------------
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)
    ax.set_box_aspect((x, y, z))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if snapshot_view is not None:
        elev, azim = snapshot_view
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=30, azim=120)

    plt.tight_layout()
    plt.show()


def plot_3d_volume_surface(volume, threshold=None, cmap="viridis", 
                           alpha=0.5, title="3D Surface Visualisation",
                           view="default"):
    """Plot 3D volume surface using marching cubes algorithm.
    
    Extracts and renders the surface mesh of a 3D volume using the marching
    cubes algorithm from scikit-image. The marching cubes algorithm identifies
    the boundary surface where the volume crosses a threshold value, creating
    a triangulated mesh representation. This approach is efficient for
    visualising volumes of any size.
    
    The function supports multiple viewing perspectives including orthogonal
    views (top, front, side) and custom angles.
    
    Parameters:
        volume (np.ndarray): 3D array of shape (Z, Y, X) containing volumetric data.
                            Can be any numeric type (int, float, etc.)
        threshold (float, optional): Isovalue for surface extraction. The surface
                                    is extracted where volume values equal this threshold.
                                    If None, automatically set to the midpoint between
                                    the minimum and maximum values in the volume.
                                    For binary volumes, typically use 0.5.
        cmap (str): Matplotlib colourmap name for surface colouring. Common options
                   include 'viridis', 'plasma', 'coolwarm', 'gray'. The colour
                   represents the Z-coordinate of the surface.
        alpha (float): Surface transparency value between 0.0 (fully transparent)
                      and 1.0 (fully opaque). Values around 0.5-0.7 work well
                      for seeing surface details.
        title (str): Title text displayed above the visualisation
        view (str or tuple): Camera viewing perspective. Options:
                            - "default": Standard 3D perspective (elevation=30°, azimuth=45°)
                            - "xy" or "top": Top-down view looking along Z-axis
                            - "xz" or "front": Front view looking along Y-axis  
                            - "yz" or "side": Side view looking along X-axis
                            - tuple (elev, azim): Custom view with elevation and azimuth
                              angles in degrees. Elevation is angle above the XY plane
                              (0-90°), azimuth is rotation around the Z-axis (0-360°)
    
    Returns:
        None: Displays an interactive matplotlib 3D figure window
    
    Example usages:
        # Extract and view surface from binary segmentation
        plot_3d_volume_surface(binary_volume, threshold=0.5, cmap="gray")
        
        # View distance transform isosurface from the side
        plot_3d_volume_surface(distance_map, threshold=10, cmap="viridis", view="side")
        
        # Custom viewing angle with specific elevation and azimuth
        plot_3d_volume_surface(volume, threshold=100, view=(45, 120))
        
        # View from above (top-down)
        plot_3d_volume_surface(volume, threshold=50, view="top")
    
    Notes:
        - Requires scikit-image for the marching cubes algorithm
        - More efficient than voxel-based rendering for large volumes
        - The resulting plot is interactive: click and drag to rotate,
          scroll to zoom, right-click and drag to pan
    
    References:
        - Lorensen & Cline (1987). "Marching cubes: A high resolution 3D
          surface construction algorithm." SIGGRAPH Computer Graphics, 21(4).
        - scikit-image documentation: https://scikit-image.org/docs/stable/api/skimage.measure.html#marching-cubes
    """

    # Determine threshold value if not provided by user
    if threshold is None:
        threshold = (volume.min() + volume.max()) / 2.0
    
    # Extract triangulated surface mesh using marching cubes algorithm
    # Returns vertices (3D coordinates), faces (triangles), normals, and values
    verts, faces, normals, values = measure.marching_cubes(
        volume, level=threshold, spacing=(1.0, 1.0, 1.0)
    )
    
    # Create figure with 3D axes for plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Render the triangulated surface mesh
    # verts[:, 2] = X coordinates, verts[:, 1] = Y coordinates, verts[:, 0] = Z coordinates
    mesh = ax.plot_trisurf(
        verts[:, 2], verts[:, 1], faces, verts[:, 0],
        cmap=cmap, alpha=alpha, linewidth=0, antialiased=True
    )
    
    # Add colourbar showing the colour scale
    fig.colorbar(mesh, ax=ax, pad=0.1, shrink=0.8)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio to prevent distortion
    # This ensures that one unit in X equals one unit in Y and Z
    ax.set_box_aspect([volume.shape[2], volume.shape[1], volume.shape[0]])
    
    # Configure camera viewing angle based on view parameter
    if isinstance(view, tuple):
        # User provided custom view angles as (elevation, azimuth)
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)
    elif view.lower() in ["xy", "top"]:
        # View from above, looking down the Z-axis
        ax.view_init(elev=90, azim=0)
    elif view.lower() in ["xz", "front"]:
        # View from the front, looking along the Y-axis
        ax.view_init(elev=0, azim=0)
    elif view.lower() in ["yz", "side"]:
        # View from the side, looking along the X-axis
        ax.view_init(elev=0, azim=90)
    elif view.lower() == "default":
        # Standard 3D perspective view
        ax.view_init(elev=30, azim=45)
    else:
        # Unrecognised view option, fallback to default
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()


def plot_3d_volume_voxels(volume, threshold_lo=None, threshold_hi=None, 
                          cmap="coolwarm", alpha=0.8, title="3D Voxel Visualisation"):
    """Plot 3D volume by rendering individual voxels.
    
    Creates a 3D visualisation showing volumetric data as discrete coloured cubes.
    Each voxel is rendered based on its intensity value. Works with both binary
    volumes (True/False) and intensity volumes (grayscale values). Supports
    rendering voxels within a specified intensity range.
    
    WARNING: Slow for large volumes (>50³ voxels).
    
    Parameters:
        volume (np.ndarray): 3D array of shape (Z, Y, X) containing volumetric data.
                            Can be binary (bool, 0/1) or intensity values (uint8, float)
        threshold_lo (float, optional): Lower threshold - only display voxels with values
                                       above or equal to this. If None, no lower limit.
        threshold_hi (float, optional): Upper threshold - only display voxels with values
                                       below or equal to this. If None, no upper limit.
        cmap (str): Matplotlib colourmap name
        alpha (float): Transparency (0.0 = invisible, 1.0 = opaque)
        title (str): Plot title
    
    Returns:
        None: Displays matplotlib figure
    
    Example usage:
        # Binary volume - all True voxels
        plot_3d_volume_voxels(binary_volume, cmap="gray")
        
        # Only voxels with intensity between 10 and 50
        plot_3d_volume_voxels(distance_map, threshold_lo=10, threshold_hi=50)
        
        # Only voxels above 20
        plot_3d_volume_voxels(image3d, threshold_lo=20, cmap="viridis")
        
        # Only voxels below 100
        plot_3d_volume_voxels(image3d, threshold_hi=100, cmap="plasma")
    """
    # Determine which voxels to display based on threshold range
    if threshold_lo is None and threshold_hi is None:
        # No thresholds specified - show all non-zero voxels
        filled = volume > 0
    elif threshold_lo is not None and threshold_hi is None:
        # Only lower threshold specified
        filled = volume >= threshold_lo
    elif threshold_lo is None and threshold_hi is not None:
        # Only upper threshold specified
        filled = volume <= threshold_hi
    else:
        # Both thresholds specified - voxels must be within range
        filled = (volume >= threshold_lo) & (volume <= threshold_hi)
    
    if not np.any(filled):
        print(f"No voxels to display within threshold range.")
        print(f"Volume range: [{volume.min()}, {volume.max()}]")
        if threshold_lo is not None:
            print(f"Lower threshold: {threshold_lo}")
        if threshold_hi is not None:
            print(f"Upper threshold: {threshold_hi}")
        return
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colourmap
    cmap_func = plt.cm.get_cmap(cmap)
    
    # Check if volume is binary (only has 0/1 or True/False values)
    unique_vals = np.unique(volume)
    is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False})
    
    if is_binary:
        # For binary volumes, use a constant colour from the colourmap
        colors = np.zeros(volume.shape + (4,), dtype=float)
        # Apply the colourmap value for "1" or "True" voxels
        colors[filled] = cmap_func(0.7)  # Use 0.7 position in colourmap
        colors[filled, 3] = alpha
    else:
        # For intensity volumes, map values to colours
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            norm_volume = (volume - vmin) / (vmax - vmin)
        else:
            norm_volume = np.ones_like(volume, dtype=float) * 0.5
        
        # Create RGBA colour array
        colors = cmap_func(norm_volume)
        
        # Set alpha: transparent for non-filled, specified alpha for filled
        colors[~filled, 3] = 0
        colors[filled, 3] = alpha
    
    # Draw voxels
    ax.voxels(filled, facecolors=colors, edgecolors=None)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()