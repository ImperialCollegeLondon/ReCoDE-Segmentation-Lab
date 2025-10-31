#!/usr/bin/env python3

# Load Python packages
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import filters

from image_processing import chamfer_distance_3d, otsu_threshold
from src.shape_creation import create_two_spheres_example


@pytest.fixture
def test_volume():
    """Provide a 3D test volume with two spheres for algorithm comparison.
    
    Creates a standardised test image containing two spherical objects suitable
    for evaluating segmentation and distance transform algorithms.
    
    Returns:
        np.ndarray: 3D array containing the test volume with two spheres.
    """
    return create_two_spheres_example()


def test_otsu_threshold_matches_skimage(test_volume):
    """Verify custom Otsu threshold implementation matches scikit-image.
    
    Compares the threshold value computed by the custom otsu_threshold function
    against scikit-image's reference implementation. Both should produce identical
    threshold values for the same input.
    
    Args:
        test_volume: 3D numpy array fixture containing test image data.
    """
    # Compute threshold using custom implementation
    custom_threshold = otsu_threshold(test_volume)
    
    # Compute threshold using scikit-image library
    library_threshold = filters.threshold_otsu(test_volume)
    
    # Verify implementations produce identical threshold values
    assert custom_threshold == library_threshold, (
        f"Custom threshold ({custom_threshold}) does not match "
        f"library threshold ({library_threshold})"
    )


def test_chamfer_distance_approximates_euclidean(test_volume):
    """Verify custom chamfer distance approximates scipy's Euclidean transform.
    
    Compares the custom chamfer distance implementation against scipy's exact
    Euclidean distance transform. The chamfer method is an integer approximation,
    so we verify it produces similar results rather than exact matches.
    
    The test checks:
    - Correlation coefficient between methods is high (> 0.95)
    - Relative error is acceptably low (< 20%)
    
    Args:
        test_volume: 3D numpy array fixture containing test image data.
    """
    # Create binary volume by thresholding
    threshold = otsu_threshold(test_volume)
    binary_volume = test_volume > threshold
    
    # Compute distance using custom chamfer implementation
    custom_distance = chamfer_distance_3d(binary_volume)
    
    # Compute distance using scipy's exact Euclidean transform
    library_distance = ndi.distance_transform_edt(binary_volume)
    
    # Verify the distance maps are highly correlated
    # Chamfer is an approximation, so we expect strong correlation rather than equality
    correlation = np.corrcoef(
        custom_distance.ravel(), 
        library_distance.ravel()
    )[0, 1]
    assert correlation > 0.95, (
        f"Distance maps poorly correlated (r={correlation:.3f}). "
        "Custom chamfer implementation may be incorrect."
    )
    
    # Verify relative error is within acceptable bounds
    # Calculate normalised RMSE to account for scale differences
    mask = library_distance > 0  # Only compare non-zero distances
    relative_error = np.sqrt(
        np.mean(
            ((custom_distance[mask] - library_distance[mask]) / 
             library_distance[mask]) ** 2
        )
    )
    assert relative_error < 0.20, (
        f"Relative error too high ({relative_error:.2%}). "
        "Chamfer approximation quality insufficient."
    )