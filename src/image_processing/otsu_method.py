#!/usr/bin/env python3
"""Otsu's method function.

Script Name:    otsu_method.py
Created:        25/11/2025
Description:

Requirements:
- Python 3.14
- Required libraries:
* numpy
"""

# -----------------------------
# Load Python packages
import numpy as np


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