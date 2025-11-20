"""Visual debugging example for 3D image processing steps."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from debugging.shapes_to_test import (
    three_spheres_all_touching,
)
from image_processing import (
    chamfer_distance_3d,
    otsu_threshold,
)
from visualisation import plot_3d_volume_surface, plot_3d_volume_voxels

image3d = three_spheres_all_touching()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Custom angle (elevation=45°, azimuth=120°)
plot_3d_volume_voxels(ax, image3d)
plt.show()
exit()

plot_3d_volume_surface(image3d, view=(45, 120))

# -----------------------------
# 2. Calculate Otsu threshold and create binary image

# Compute optimal threshold value using Otsu's method
thresh_value = otsu_threshold(image3d)
print(f"Otsu threshold value: {thresh_value}")

# Apply threshold to create binary segmentation
binary_volume = image3d > thresh_value
plot_3d_volume_voxels(binary_volume, cmap="gray")
plot_3d_volume_surface(binary_volume, view=(45, 120))
print(f"Binary volume contains {np.count_nonzero(binary_volume)} foreground voxels")

# -----------------------------
# 3. Apply distance transform

# Compute chamfer distance transform on binary volume
# Distance map shows distance from each background voxel to nearest foreground
distance_map = chamfer_distance_3d(binary_volume)

plot_3d_volume_voxels(distance_map, threshold_hi=3)
plot_3d_volume_voxels(distance_map)
print(f"Our distance transform complete. Max distance: {np.max(distance_map)}")

library_distance = ndi.distance_transform_edt(binary_volume)
plot_3d_volume_voxels(library_distance, threshold_lo=0.1)
plot_3d_volume_voxels(library_distance)
print(f"library distance transform complete. Max distance: {np.max(library_distance)}")
