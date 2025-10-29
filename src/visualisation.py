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
