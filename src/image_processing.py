#!/usr/bin/env python3
"""Image processing functions.

Script Name:    image_processing.py
Author:         David Buchner, Imperial College London
Created:        22/10/2025
Last Modified:  22/10/2025
Description:
Functions to process/manipulate 3d numpy arrays (image)

Requirements:
- Python 3.14
- Required libraries:
* numpy
"""

# -----------------------------
# Load Python packages
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
import time

# -----------------------------


def otsu_threshold(array3d):
    """Compute the Otsu threshold.

    Compute the Otsu threshold from a 3D NumPy array of integer voxel intensities.

    This function implements Otsu's method, matching the behavior of scikit-image's
    `threshold_otsu`. It is suitable for integer-valued arrays such as uint8 or uint16,
    commonly used in image stacks.

    Mathematical background:
    Otsu's method selects the threshold that maximizes the between-class variance 
    sigma_b^2(t), which quantifies how well the image is separated into foreground and
    background.

    This function uses the following vectorized formulation:
        sigma_b^2(t) = [mu_T · omega(t) - mu(t)]^2 / [omega(t) · (1 - omega(t))]
    where:
        - omega(t): cumulative probability of pixels below threshold t
        - mu(t): cumulative mean of pixels below threshold t
        - mu_T: global mean of all pixel intensities

    This is mathematically equivalent to the more common form:
        sigma_b^2(t) = omega_b · omega_f · (mu_b - mu_f)^2
    where:
        - omega_b, omega_f: probabilities of background and foreground classes
        - mu_b, mu_f: mean intensities of background and foreground classes

    References:
    - Wikipedia overview: https://en.wikipedia.org/wiki/Otsu%27s_method
    - Chityala, Ravishankar, and Sridevi Pudipeddi. "Image Processing and Acquisition 
      Using Python. Second edition." Boca Raton: Chapman & Hall/CRC, 2020. Print.
    - Original paper: Otsu, N. (1979). "A threshold selection method from gray-level 
      histograms." IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.

    Parameters:
        array3d (np.ndarray): 3D array of shape (depth, height, width)
        containing integer voxel values.

    Returns:
        int: Otsu threshold value that separates foreground from background.

    Example usage:
        threshold = otsu_threshold(stack)
    """
    # Flatten the 3D volume into a 1D array of voxel values
    voxels = array3d.ravel()

    # Ensure the input is integer-based (Otsu assumes discrete intensity levels)
    if not np.issubdtype(voxels.dtype, np.integer):
        raise ValueError("Input must be integer-valued")

    # Get the maximum possible value for the data type
    # (e.g., 255 for uint8, 65535 for uint16 (typical for micro CT data))
    max_val = np.iinfo(voxels.dtype).max

    # Compute the histogram using bincount (fast for integer arrays)
    hist = np.bincount(voxels, minlength=max_val + 1).astype(float)

    # Normalize the histogram to get a probability distribution
    prob = hist / hist.sum()

    # Compute cumulative probability (ω) and cumulative mean (μ)
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(len(prob)))

    # Compute global mean (μ_T): mean intensity of the entire image
    mean_total = cum_mean[-1]

    # Compute between-class variance for each possible threshold
    # This measures how well a threshold separates the image into two classes.
    variance = np.divide(
        (mean_total * cum_prob - cum_mean) ** 2,
        cum_prob * (1 - cum_prob),
        out=np.zeros_like(cum_prob),
        where=(cum_prob * (1 - cum_prob)) != 0,
    )

    return np.argmax(variance)


def chamfer_distance_3d(img):
    """Compute the chamfer distance transform.

    Compute the chamfer distance transform of a 3D binary NumPy array.

    This function approximates the Euclidean distance transform using a chamfer mask,
    which propagates integer-valued distances through local neighborhoods. It is 
    suitable for binary 3D arrays where foreground voxels are non-zero and background 
    voxels are zero.

    Mathematical background:
    The chamfer method uses a fixed set of weighted offsets to simulate Euclidean 
    distances through iterative forward and backward passes. Each voxel updates its 
    value based on neighboring voxels and predefined weights, approximating the 
    shortest path to the nearest foreground voxel.

    This implementation uses a 3x3x3 mask with integer weights:
        - 3 for axis-aligned neighbors
        - 4 for face-diagonal neighbors
        - 5 for corner-diagonal neighbors

    The algorithm consists of two phases:
        - Forward pass: propagates distances from top-left-front to 
          bottom-right-back
        - Backward pass: refines distances from bottom-right-back to 
          top-left-front

    This method is significantly faster than exact Euclidean transforms and is well-
    suited for large volumes where performance is critical.

    References:
    - Borgefors, G. (1986). "Distance transformations in digital images." 
      Computer Vision, Graphics, and Image Processing, 34(3), 344-371.
    - Wikipedia overview: https://en.wikipedia.org/wiki/Distance_transform#Chamfer_distance_transform
    - Stack Overflow discussion on NumPy-only distance transform performance: 
      https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python

    Parameters:
        img (np.ndarray): 3D binary array of shape (X, Y, Z) with 0 for background
                          and non-zero for foreground.

    Returns:
        np.ndarray: 3D array of same shape with chamfer distances (integer 
                    approximation of Euclidean distance).

    Example usage:
        distance_map = chamfer_distance_3d(binary_volume)
    """
    # Start timing
    start_time = time.time()

    # Define chamfer weights (3x3x3 neighborhood)
    weights = [
        (1, 0, 0, 3),
        (0, 1, 0, 3),
        (0, 0, 1, 3),
        (1, 1, 0, 4),
        (1, 0, 1, 4),
        (0, 1, 1, 4),
        (1, 1, 1, 5),
    ]

    shape = img.shape
    sx, sy, sz = shape

    # Initialize distance map: background gets large value, foreground gets
    dt = np.where(img == 0, np.max(img.shape), 0).astype(np.uint32)

    # Forward pass: iterate from top-left-front to bottom-right-back
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if img[x, y, z] == 0:
                    for dx, dy, dz, w in weights:
                        nx, ny, nz = x - dx, y - dy, z - dz
                        if (
                            0 <= nx < shape[0]
                            and 0 <= ny < shape[1]
                            and 0 <= nz < shape[2]
                        ):
                            dt[x, y, z] = min(dt[x, y, z], dt[nx, ny, nz] + w)

    # Backward pass: iterate from bottom-right-back to top-left-front
    for z in reversed(range(shape[2])):
        for y in reversed(range(shape[1])):
            for x in reversed(range(shape[0])):
                if img[x, y, z] == 0:
                    for dx, dy, dz, w in weights:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (
                            0 <= nx < shape[0]
                            and 0 <= ny < shape[1]
                            and 0 <= nz < shape[2]
                        ):
                            dt[x, y, z] = min(dt[x, y, z], dt[nx, ny, nz] + w)

    elapsed = time.time() - start_time
    print(f"Distance transform: {elapsed:.4f} seconds.")

    return dt


def chamfer_distance_3d_structured(img):
    """Compute the chamfer distance transform with structured arrays.

    Chamfer distance transform using structured arrays and sorted voxel iteration.
    Performs single sweep (forward and backward) with integer-weighted neighbors,
    image padded to avoid if statement.

    Parameters:
        img (np.ndarray): 3D binary array (non-zero = foreground, 0 = background)

    Returns:
        np.ndarray: Chamfer distance map (same shape as input)
    """
    start_time = time.time()

    # Define symmetric chamfer mask: neighbor offsets and weights
    offsets = [
        (1, 0, 0, 3), (-1, 0, 0, 3),
        (0, 1, 0, 3), (0, -1, 0, 3),
        (0, 0, 1, 3), (0, 0, -1, 3),
        (1, 1, 0, 4), (-1, -1, 0, 4),
        (1, 0, 1, 4), (-1, 0, -1, 4),
        (0, 1, 1, 4), (0, -1, -1, 4),
        (1, 1, 1, 5), (-1, -1, -1, 5)
    ]

    # Pad the input volume with a 1-voxel border
    padded = np.pad(img == 0, pad_width=1, mode='constant', constant_values=0)
    shape = np.array(img.shape)

    # Initialize distance map: foreground = 0, background = large value
    dt = np.where(padded, 65535, 0).astype(np.uint32)

    # Get indices of background voxels in original image space
    indices = np.argwhere(img == 0)
    indices += 1  # shift to match padded coordinates

    # Convert to structured array for sorting
    dtype = [("x", int), ("y", int), ("z", int)]
    structured_indices = unstructured_to_structured(indices, dtype=dtype)
    structured_indices = np.sort(structured_indices, order=["z", "y", "x"])

    # Combined sweep without boundary checks
    for idx in structured_indices:
        x, y, z = idx["x"], idx["y"], idx["z"]
        for dx, dy, dz, w in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            dt[z, y, x] = min(dt[z, y, x], dt[nz, ny, nx] + w)

    elapsed = time.time() - start_time
    print(f"Chamfer distance (structured) completed in {elapsed:.4f} seconds.")

    return dt[1:-1, 1:-1, 1:-1]


def chamfer_distance_3d_argwhere(img):
    """Compute the chamfer distance transform with argwhere.

    Chamfer distance transform using argwhere..
    Performs forward and backward sweeps with integer-weighted neighbors.
    Includes timing output for performance analysis.

    Parameters:
        img (np.ndarray): 3D binary array (non-zero = foreground, 0 = background)

    Returns:
        np.ndarray: Chamfer distance map (same shape as input)
    """
    start_time = time.time()
    # Define chamfer mask: neighbor offsets and weights
    neighbours = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    weights = np.array([3, 3, 3, 4, 4, 4, 5])
    shape = np.array(img.shape)
    sx, sy, sz = img.shape

    # Initialize distance map: background = 65535, foreground = 0
    dt = np.where(img == 0, 65535, 0).astype(np.uint32)
    # Get background voxel coordinates
    coords = np.argwhere(img == 0)
    # Sort coordinates in ascending z, y, x order for forward sweep
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]

    # Forward sweep
    for pos in coords:
        x, y, z = pos
        for offset, w in zip(neighbours, weights):
            nx, ny, nz = x - offset[0], y - offset[1], z - offset[2]
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                dt[x, y, z] = min(dt[x, y, z], dt[nx, ny, nz] + w)


    # Backward sweep
    for pos in coords[::-1]:
        x, y, z = pos
        for offset, w in zip(neighbours, weights):
            nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                dt[x, y, z] = min(dt[x, y, z], dt[nx, ny, nz] + w)


    elapsed = time.time() - start_time
    print(f"Chamfer distance (argwhere) completed in {elapsed:.4f} seconds.")

    return dt


def chamfer_distance_3d_optimized(img):
    """Compute the chamfer distance transform using Numpy sweeps.

    Compute the chamfer distance transform of a 3D binary NumPy array
    using an optimized NumPy approach.

    This function approximates the Euclidean distance transform using a chamfer mask,
    which propagates integer-valued distances through local neighborhoods. It is
    suitable for binary 3D arrays where foreground voxels are non-zero and background
    voxels are zero.

    This NumPy-only chamfer distance transform uses np.roll to simulate neighbor
    propagation efficiently. While not part of the classical Borgefors
    implementation, this technique is inspired by stencil-based methods and
    array-shifting strategies commonly used in high-performance computing
    and image processing. See Sling Academy for np.roll examples.

    Parameters:
        img (np.ndarray): 3D binary array of shape (X, Y, Z) with 0 for background
                          and non-zero for foreground.

    Returns:
        np.ndarray: 3D array of same shape as `img`, where each voxel contains an
                    integer-valued approximation of the Euclidean distance to the
                    nearest foreground voxel.

    ------------------------------------------------------------------------
    Example Usage:
        distance_map = chamfer_distance_3d_optimized(binary_volume)

    ------------------------------------------------------------------------
    References:
    - Borgefors, G. (1986). "Distance transformations in digital images."
      Computer Vision, Graphics, and Image Processing, 34(3), 344-371.
    - Wikipedia: https://en.wikipedia.org/wiki/Distance_transform#Chamfer_distance_transform
    - Stack Overflow: https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python
    - https://www.slingacademy.com/article/understanding-numpy-roll-function-6-examples/
    """
    # Start timing
    start_time = time.time()

    # Define chamfer mask: each tuple is (dx, dy, dz, weight)
    # These represent relative neighbor positions and their associated movement cost.
    # Includes both positive and negative directions to ensure symmetric propagation.
    offsets = [
        (1, 0, 0, 1.0), (-1, 0, 0, 1.0),
        (0, 1, 0, 1.0), (0, -1, 0, 1.0),
        (0, 0, 1, 1.0), (0, 0, -1, 1.0),
        (1, 1, 0, 1.414), (-1, -1, 0, 1.414),
        (1, 0, 1, 1.414), (-1, 0, -1, 1.414),
        (0, 1, 1, 1.414), (0, -1, -1, 1.414),
        (1, 1, 1, 1.732), (-1, -1, -1, 1.732),
    ]

    # Pad the input volume with a 1-voxel border to simplify boundary handling.
    # Padding ensures that neighbor access won't go out of bounds.
    padded = np.pad(img != 0, pad_width=1, mode='constant', constant_values=0)

    # Initialize the distance map:
    # Foreground voxels (non-zero in original image) get distance 0.
    # Background voxels (zero in original image) get a large initial value.
    dt = np.where(padded, 65535, 0).astype(np.uint32)

    # Iterative sweeping: repeat the neighbor propagation multiple times.
    # Each sweep allows distances to propagate further through the volume.
    max_iter = max(img.shape)  # Dynamic upper bound based on volume shape
    for i in range(max_iter):
        prev_dt = dt.copy()

        for dx, dy, dz, w in offsets:
            # Shift the entire distance map by (dx, dy, dz) to simulate neighbor access.
            # This gives the neighbor values for every voxel in one operation.
            shifted = np.roll(dt, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Update each voxel with the minimum of its current value and the neighbor's value + weight.
            # This is the core of the chamfer propagation: finding shorter paths via neighbors.
            dt = np.minimum(dt, shifted + w)

        # Check for convergence: if no values changed, break early
        if np.array_equal(dt, prev_dt):
            elapsed = time.time() - start_time
            print(f"Converged after {i+1} iterations. Time Distance Transform:  {elapsed:.4f} seconds.")
            break
    else:
        elapsed = time.time() - start_time
        print(f"Reached max_iter={max_iter} without full convergence. Time Distance Transform: {elapsed:.4f} seconds.")

    # Remove the padding to return a result of the original shape.
    return dt[1:-1, 1:-1, 1:-1]

