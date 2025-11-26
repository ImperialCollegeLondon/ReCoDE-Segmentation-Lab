#!/usr/bin/env python3
"""Execution and comparison of both image processing pipelines.

Author:         David Buchner, Imperial College London
Created:        31/10/2025
Last Modified:  31/10/2025
Description:
   Script to execute the image segmentation pipeline.
   Compares results and performance of the self-build
   pipeline steps against the ones availbe in libraries.

Requirements:
   - Python 3.11
   - Required libraries:
       * numpy
       *scipy
       *skimage

"""

# -----------------------------
# Load Python packages
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.segmentation import watershed

# Functions for image processing
from image_processing.distance_transform import chamfer_distance_transform
from image_processing.local_extrema import find_local_minima
from image_processing.otsu_method import otsu_threshold
from image_processing.watershed_segmentation import watershed_3d

# Import custom modules
# Functions to create synthetic 3D data
from shape_creation import create_n_spheres_example

# Functions for visualisations
from visualisation import (
    plot_2d_slice_with_values,
    plot_3d_volume_voxels,
    plot_panels,
)

# -----------------------------
# 1. Load the 3D image/dataset onto a numpy array
# i.e. into a format that allows efficient manipulation in the later steps
# Two clearly seperated spheres
# image3d = create_two_spheres_example(
#     centre1=(6, 6, 6), radius1=2, centre2=(2, 2, 2), radius2=1
# )
# Two overlapping spheres example
# image3d = create_n_spheres_example(
#     centres=[(6, 4, 5),(4, 2, 3)], radii=[2,2], intensities=[180,200],volume_size=10
# )

image3d = create_n_spheres_example(
    centres=[(8, 9, 8),(7, 6, 4)], radii=[3,3], intensities=[180,200],volume_size=15
)

# Print a statement showing the crop
print(f"3D image size {np.shape(image3d)}")

# -----------------------------
# 3. Calculate Otsu threshold and binary image
# Compute Otsu's threshold
otsu_value = otsu_threshold(image3d)
otsu_value_lib = filters.threshold_otsu(image3d)

print(f"Otsu threshold value Own: {otsu_value} Libraries: {otsu_value_lib}")

# Apply threshold
binary = image3d > otsu_value  # returns a boolean array
binary_lib = image3d > otsu_value_lib  # returns a boolean array

# -----------------------------
# 4. Apply distance transform
# Compute distance transform
starttime = time.time()
DT = chamfer_distance_transform(binary)
time_chamfer = time.time() - starttime
starttime = time.time()
DT_lib = ndi.distance_transform_edt(binary_lib)
time_scipy = starttime - time.time()
print(
    f"Computation time Chamfer Distance: {np.round(time_chamfer, 2)} seconds,"
    f" Exact Euclidean Distance (scipy): {np.round(time_scipy, 2)}seconds"
)

# Plot 3d voxel render of distance transform
# plot_two_panels(DT, DT_lib,
#                 plot_func=plot_3d_volume_voxels,projection='3d',
#                 plot_kwargs1={'threshold_lo': 1,'threshold_hi': 3},
#                 plot_kwargs2={'threshold_lo': 1,'threshold_hi': 3},
#                 title1='Chamfer distance',title2='Exact Euclidean distance')


plot_panels(
    n=2,
    data_list=[DT, DT_lib],
    plot_func=plot_2d_slice_with_values,
    plot_kwargs_list=[{"slice_index": 5}, {"slice_index": 5}],
    title="Distance from background $Z$ [-]",
    subtitles=["Chamfer distance", "Exact Euclidean distance"],
)

# -----------------------------
# 4. Determine local minimas
local_minima = find_local_minima(-DT)
local_minima_lib = morphology.local_minima(-DT_lib)

print(
    f"Nr. of local minima: {np.shape(np.argwhere(local_minima))[0]}"
    f" (scipy): {np.shape(np.argwhere(local_minima_lib))[0]}"
)

# -----------------------------
# Apply Watershed segmentation
watershed_build = watershed_3d(DT, binary, local_minima)

# Label markers
markers_lib = measure.label(local_minima_lib)
# Apply watershed
watershed_lib = watershed(-DT_lib, markers_lib, mask=binary_lib)

# Plot 3d voxel render of watershed segmentation
plot_panels(
    n=2,
    data_list=[watershed_build[-1], watershed_lib],
    plot_func=plot_3d_volume_voxels,
    plot_kwargs_list=[
        {"threshold_lo": 1, "threshold_hi": 2},
        {"threshold_lo": 1, "threshold_hi": 2},
    ],
    title=None,  # no overall title in original call
    subtitles=["Build", "Libraries"],
    projection="3d",
)

# plot_two_panels(watershed_build[2], watershed_build[2],
#                 plot_func=plot_2d_slice_with_values,
#                 plot_kwargs1={'slice_index': 5},
#                 plot_kwargs2={'slice_index': 6},
#                 title1='Chamfer distance',title2='Exact Euclidean distance')
