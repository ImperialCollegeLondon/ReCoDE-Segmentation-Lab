#!/usr/bin/env python3
"""Functions for visualisation of the 3D dataset.

Script Name:    visualize_image.py
Author:         David Buchner, Imperial College London
Created:        16/10/2025
Last Modified:  17/12/2025
Description:
Functions to visualize images in 2D and 3D, including
interactive display and metadata extraction.

Requirements:
- Python 3.x
- Required libraries:
* numpy
* matplotlib
* plotly (for 3D visualisation)

Notes:
Adapted from previous dataset loading scripts to provide
versatile image visualisation tools.

"""

# -----------------------------
# Standard library
import matplotlib

# Set backend before importing pyplot
matplotlib.use("TkAgg")  # or 'Qt5Agg'

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# -----------------------------
# Functions that create the matplotlib visualisations
# Subfunction of the 3D voxel based volume render


def _compute_filled_mask(
    volume: np.ndarray, threshold_lo: float | None, threshold_hi: float | None
) -> np.ndarray:
    """Return boolean mask of voxels to display based on threshold range.

    Parameters:
            volume (np.ndarray): 3D array of shape (Z, Y, X) containing volumetric data.
                                 Can be binary (bool) or intensity values (int, float).
            threshold_lo (float | None): Lower threshold. Voxels with
                                         values >= threshold_lo
                                         are considered filled. If None, no lower bound.
            threshold_hi (float | None): Upper threshold. Voxels with
                                         values <= threshold_hi
                                         are considered filled. If None, no upper bound.

    Returns:
            np.ndarray: Boolean mask of shape (Z, Y, X) indicating which voxels are
                        filled under the specified threshold rules. If both thresholds
                        are None, voxels with values > 0 are considered filled.

    """
    if threshold_lo is None and threshold_hi is None:
        return volume > 0
    elif threshold_lo is not None and threshold_hi is None:
        return volume >= threshold_lo
    elif threshold_lo is None and threshold_hi is not None:
        return volume <= threshold_hi
    else:
        return (volume >= threshold_lo) & (volume <= threshold_hi)


def _build_discrete_colormap(
    labels: np.ndarray, cmap_func
) -> tuple[ListedColormap, BoundaryNorm]:
    """Build a discrete ListedColormap and BoundaryNorm for categorical labels.

    Boundaries are placed halfway between consecutive label values. For a single label,
        a small bin is created around it.

    Parameters:
        labels (np.ndarray): 1D array of unique label values (e.g., int or float),
                             typically extracted from volume[filled].
        cmap_func (matplotlib.colors.Colormap): Base colormap function obtained via
                                                `plt.cm.get_cmap(name)`.

    Returns:
        tuple[ListedColormap, BoundaryNorm]:
            - ListedColormap: Discrete colormap with one color per label.
            - BoundaryNorm: Normalization mapping label ranges to colormap indices.

    """
    labels = np.sort(labels)
    n_labels = len(labels)

    # Evenly sample colors from the base colormap
    color_list = [cmap_func(i / max(1, n_labels - 1)) for i in range(n_labels)]
    listed_cmap = ListedColormap(color_list)

    if n_labels == 1:
        boundaries = [labels[0] - 0.5, labels[0] + 0.5]
    else:
        mids = (labels[:-1] + labels[1:]) / 2.0
        boundaries = [
            2 * labels[0] - mids[0],
            *mids,
            2 * labels[-1] - mids[-1],
        ]

    norm = BoundaryNorm(boundaries, listed_cmap.N)
    return listed_cmap, norm


def _map_colors_discrete(
    volume: np.ndarray,
    filled: np.ndarray,
    listed_cmap: ListedColormap,
    norm: BoundaryNorm,
    alpha: float,
) -> np.ndarray:
    """Map all voxels to RGBA using discrete colormap/norm; apply alpha to filled only.

    Transparency is applied only to voxels marked as filled; non-filled voxels are
    made fully transparent.

    Parameters:
        volume (np.ndarray): 3D volume (Z, Y, X) with label values to map to colors.
        filled (np.ndarray): Boolean mask (Z, Y, X) indicating voxels to display.
        listed_cmap (ListedColormap): Discrete colormap (one color per label).
        norm (BoundaryNorm): Normalizer that maps label values into colormap indices.
        alpha (float): Opacity applied to filled voxels (0.0-1.0).

    Returns:
        np.ndarray: RGBA color array of shape (Z, Y, X, 4) with alpha set to `alpha`
                    for filled voxels and 0.0 for non-filled voxels

    """
    colors = np.zeros((*volume.shape, 4), dtype=float)
    mapped_colors = listed_cmap(norm(volume))
    colors[...] = mapped_colors
    colors[~filled, 3] = 0.0
    colors[filled, 3] = alpha
    return colors


def _map_colors_continuous(
    volume: np.ndarray, filled: np.ndarray, cmap_func, alpha: float
) -> tuple[np.ndarray, float, float]:
    """Map intensities to RGBA continuously and apply transpa.

    Uses linear normalization between the global min/max of the volume. If the volume
    is constant (vmax == vmin), a mid-tone color is used. Alpha is applied to filled
    voxels; non-filled voxels are transparent.

    Parameters:
        volume (np.ndarray): 3D volume (Z, Y, X) with continuous intensity values.
        filled (np.ndarray): Boolean mask (Z, Y, X) indicating voxels to display.
        cmap_func (matplotlib.colors.Colormap): Base colormap function.
        alpha (float): Opacity applied to filled voxels (0.0-1.0).

    Returns:
        tuple[np.ndarray, float, float]:
            - np.ndarray: RGBA color array of shape (Z, Y, X, 4).
            - float: vmin used for normalization.
            - float: vmax used for normalization.

    """
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax > vmin:
        norm_volume = (volume - vmin) / (vmax - vmin)
    else:
        norm_volume = np.ones_like(volume, dtype=float) * 0.5  # flat color if constant

    rgba = cmap_func(norm_volume)
    colors = np.zeros((*volume.shape, 4), dtype=float)
    colors[...] = rgba
    colors[~filled, 3] = 0.0
    colors[filled, 3] = alpha
    return colors, vmin, vmax


def _add_colorbar_discrete(
    ax, listed_cmap: ListedColormap, norm: BoundaryNorm, labels: np.ndarray
):
    """Create a discrete colorbar with ticks at the label values.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes to attach the colorbar.
        listed_cmap (ListedColormap): Discrete colormap used for the plot.
        norm (BoundaryNorm): Normalizer used to map labels to colormap indices.
        labels (np.ndarray): Sorted unique label values displayed in the plot.

    Returns:
        matplotlib.colorbar.Colorbar: The created colorbar instanc

    """
    sm = plt.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_ticks(labels)
    cbar.set_ticklabels([str(v) for v in labels])
    cbar.set_label("Labels")
    return cbar


def _add_colorbar_continuous(ax, cmap_func, vmin: float, vmax: float):
    """Create a continuous colorbar mapped to a value range [vmin, vmax].

    Parameters:
        ax (matplotlib.axes.Axes): Target axes to attach the colorbar.
        cmap_func (matplotlib.colors.Colormap): Colormap used for continuous mapping.
        vmin (float): Minimum value for color normalization.
        vmax (float): Maximum value for color normalization.

    Returns:
        matplotlib.colorbar.Colorbar: The created colorbar instance.
    """
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_func, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Intensity")
    return cbar


def plot_3d_volume_voxels(
    ax,
    volume: np.ndarray,
    threshold_lo: float | None = None,
    threshold_hi: float | None = None,
    cmap: str = "tab20",
    alpha: float = 0.8,
    title: str | None = None,
    force_discrete: bool = True,
) -> None:
    """Plot 3D volume by rendering individual voxels.

    Creates a 3D visualization showing volumetric data as discrete colored cubes.
    Each voxel is rendered based on its intensity/label value. Works with both binary
    volumes (True/False) and intensity volumes (grayscale values). Supports rendering
    voxels within a specified intensity range.

    WARNING: Slow for large volumes (> ~50^3 voxels).

    Parameters:
        ax (matplotlib.axes.Axes): A 3D axes (e.g., `projection="3d"`) to draw into.
        volume (np.ndarray): 3D array of shape (Z, Y, X). May be binary or continuous.
        threshold_lo (float | None): Lower threshold; display voxels with values
                                     >= threshold_lo. If None, no lower limit.
        threshold_hi (float | None): Upper threshold; display voxels with values
                                     <= threshold_hi. If None, no upper limit.
        cmap (str): Matplotlib colormap name (e.g., "tab20", "viridis").
        alpha (float): Transparency (0.0 = invisible, 1.0 = opaque) for filled voxels.
        title (str | None): Optional plot title.
        force_discrete (bool): If True, use discrete mapping (labels). If False and
                               volume is not binary, use continuous mapping.

    Returns:
        None: Draws the voxels and a colorbar into the provided axes.
    """
    # 1) Decide which voxels to show
    filled = _compute_filled_mask(volume, threshold_lo, threshold_hi)
    if not np.any(filled):
        # Print diagnostics when no voxels are within the threshold range.
        print("No voxels to display within threshold range.")
        print(f"Volume range: [{volume.min()}, {volume.max()}]")
        if threshold_lo is not None:
            print(f"Lower threshold: {threshold_lo}")
        if threshold_hi is not None:
            print(f"Upper threshold: {threshold_hi}")
        return

    # 2) Base colormap function
    cmap_func = plt.cm.get_cmap(cmap)

    # 3) Binary check and label extraction in the filled region
    # --- Determine if binary ---
    unique_vals_all = np.unique(volume)
    is_binary = (len(unique_vals_all) <= 2) and set(unique_vals_all).issubset(
        {0, 1, True, False}
    )

    # Compute unique labels within the filled region
    unique_vals_filled = np.unique(volume[filled])

    # 4) Choose discrete vs continuous
    use_discrete = force_discrete or is_binary

    # 5) Color mapping + draw + colorbar
    if use_discrete:
        listed_cmap, norm = _build_discrete_colormap(unique_vals_filled, cmap_func)
        colors = _map_colors_discrete(volume, filled, listed_cmap, norm, alpha)
        # Render the voxel grid.
        ax.voxels(filled, facecolors=colors, edgecolors=None)
        # Add discrete colour bar
        _add_colorbar_discrete(ax, listed_cmap, norm, unique_vals_filled)
    else:
        colors, vmin, vmax = _map_colors_continuous(volume, filled, cmap_func, alpha)
        # Render the voxel grid.
        ax.voxels(filled, facecolors=colors, edgecolors=None)
        # Add continuous colour bar
        _add_colorbar_continuous(ax, cmap_func, vmin, vmax)

    # 6) Axis labels + optional title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set title
    if title:
        ax.set_title(title)


# -----------------------------
# FOther rendering functions.


def plot_2d_slice_with_values(
    ax, volume, axis=0, slice_index=None, cmap="viridis", title=None
):
    """Plot a 2D slice from a 3D volume with a grid and voxel values.

    Parameters:
        volume (np.ndarray): 3D numpy array.
        axis (int): Axis normal to the slice (0, 1, or 2).
        slice_index (int, optional): Index of the slice to extract.
        If None, uses the central slice.
        cmap (str): Matplotlib colormap for background shading.

    Returns:
        None
    """
    assert volume.ndim == 3, "Input volume must be 3D"
    assert axis in (0, 1, 2), "Axis must be 0, 1, or 2"

    # Determine slice index
    if slice_index is None:
        slice_index = volume.shape[axis] // 2

    # Extract the 2D slice
    if axis == 0:
        slice_2d = volume[slice_index, :, :]
    elif axis == 1:
        slice_2d = volume[:, slice_index, :]
    else:
        slice_2d = volume[:, :, slice_index]

    ax.imshow(slice_2d, cmap=cmap, origin="upper")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, slice_2d.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, slice_2d.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    # Annotate voxel values
    for i in range(slice_2d.shape[0]):
        for j in range(slice_2d.shape[1]):
            val = slice_2d[i, j]
            ax.text(
                j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8
            )


def plot_hist(voxels, bins=None, t=None, title="Histogram with threshold"):
    """Plot a histogram of voxel intensities with an optional threshold marker.

    Creates a 1D histogram of voxel intensity values from a 3D volume or flattened
    array. Optionally overlays a vertical line at a specified threshold value.

    Parameters:
        voxels (np.ndarray): 1D or flattened array of voxel intensity values.
                             Typically obtained from a 3D volume using `.ravel()`.
        bins (int | None): Number of histogram bins. If None, defaults to the full
                           range of the voxel data type (e.g., np.iinfo(dtype).max + 1).
        t (float | None): Optional threshold value to mark on the histogram with a
                          vertical dashed line. If None, no threshold line is drawn.
        title (str): Title for the plot. Defaults to "Histogram with threshold".

    Returns:
        None: Displays a matplotlib figure with the histogram
        and optional threshold line.
    """
    bins = bins or (np.iinfo(voxels.dtype).max + 1)
    hist = np.bincount(voxels, minlength=bins).astype(float)
    xs = np.arange(len(hist))
    plt.figure(figsize=(6, 3.5))
    plt.plot(xs, hist, color="steelblue", lw=1.5)
    if t is not None:
        plt.axvline(t, color="crimson", ls="--", lw=2, label=f"t = {t}")
        plt.legend()
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Function to create figures(subplot slots) which are filled with rendering functions.


def plot_panels(
    n,
    data_list,
    plot_func,
    plot_kwargs_list=None,
    title=None,
    subtitles=None,
    figsize=(16, 8),
    layout=None,
    projection=None,
):
    """Function to create a figure with n subplots.

    Parameters:
        data_list: List of datasets to plot.
        plot_func: Function that accepts an Axes object and a dataset,
        plus optional kwargs.
        n: Number of subplots/panels.
        plot_kwargs_list: List of dictionaries of kwargs for plot_func.
        titles: List of titles for each subplot.
        figsize: Size of the overall figure.
        layout: Tuple indicating subplot layout (rows, cols). If None, auto layout
          is used.
        projection: Projection for all subplots ('3d' or None).

    Returns:
        None: Displays the matplotlib figure.
    """
    if plot_kwargs_list is None:
        plot_kwargs_list = [{} for _ in range(n)]
    if subtitles is None:
        subtitles = [None] * n
    if layout is None:
        # Auto layout: try to make it roughly square
        rows = int(n**0.5)
        cols = (n + rows - 1) // rows
        layout = (rows, cols)

    fig, axes = plt.subplots(
        layout[0],
        layout[1],
        figsize=figsize,
        subplot_kw={"projection": projection} if projection else {},
    )

    # Ensure axes is always a flat list
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n):
        plot_func(axes[i], data_list[i], **plot_kwargs_list[i])
        if subtitles[i]:
            axes[i].set_title(subtitles[i], fontsize=14)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=20)

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)
    plt.show()
