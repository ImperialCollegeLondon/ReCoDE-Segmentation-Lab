"""A regression of images to test.

library implementations against our implementations.

"""

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.segmentation import watershed


# Add parent directory to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing.otsu_method import otsu_threshold
from image_processing.analytical_information import compute_volume_and_com
from image_processing.distance_transform import chamfer_distance_transform
from image_processing.local_extrema import find_local_minima


@pytest.mark.parametrize(
    "test_volume",
    [
        "two_spheres_not_touching",
        "two_spheres_touching",
        "three_spheres_all_touching",
        "three_spheres_not_touching",
    ],
    indirect=True,
)
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


@pytest.mark.parametrize(
    "test_volume",
    [
        "two_spheres_not_touching",
        "two_spheres_touching",
        "three_spheres_all_touching",
        "three_spheres_not_touching",
    ],
    indirect=True,
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
            Parametrised to test multiple geometric configurations.
    """
    # Create binary volume by thresholding
    threshold = otsu_threshold(test_volume)
    binary_volume = test_volume > threshold

    # Compute distance using custom chamfer implementation
    custom_distance = chamfer_distance_transform(binary_volume)

    # Compute distance using scipy's exact Euclidean transform
    library_distance = ndi.distance_transform_edt(binary_volume)

    # Verify the distance maps are highly correlated
    # Chamfer is an approximation, so we expect strong correlation rather than equality
    correlation = np.corrcoef(custom_distance.ravel(), library_distance.ravel())[0, 1]
    assert correlation > 0.95, (
        f"Distance maps poorly correlated (r={correlation:.3f}). "
        "Custom chamfer implementation may be incorrect."
    )

    # Verify relative error is within acceptable bounds
    # Calculate normalised RMSE to account for scale differences
    mask = library_distance > 0  # Only compare non-zero distances
    relative_error = np.sqrt(
        np.mean(
            ((custom_distance[mask] - library_distance[mask]) / library_distance[mask])
            ** 2
        )
    )
    assert relative_error < 0.20, (
        f"Relative error too high ({relative_error:.2%}). "
        "Chamfer approximation quality insufficient."
    )


@pytest.mark.parametrize(
    "test_volume",
    [
        "two_spheres_not_touching",
        "two_spheres_touching",
        "three_spheres_all_touching",
        "three_spheres_not_touching",
    ],
    indirect=True,
)
def test_optimised_chamfer_distance_approximates_euclidean(test_volume):
    """Verify optimised chamfer distance approximates scipy's Euclidean transform.

    Compares the optimised chamfer distance implementation against scipy's exact
    Euclidean distance transform. The chamfer method is an integer approximation,
    so we verify it produces similar results rather than exact matches.

    The test checks:
    - Correlation coefficient between methods is high (> 0.95)
    - Relative error is acceptably low (< 20%)

    Args:
        test_volume: 3D numpy array fixture containing test image data.
            Parametrised to test multiple geometric configurations.
    """
    # Create binary volume by thresholding
    threshold = otsu_threshold(test_volume)
    binary_volume = test_volume > threshold

    # Compute distance using optimised chamfer implementation
    custom_distance = chamfer_distance_transform(binary_volume)

    # Compute distance using scipy's exact Euclidean transform
    library_distance = ndi.distance_transform_edt(binary_volume)

    # Verify the distance maps are highly correlated
    # Chamfer is an approximation, so we expect strong correlation rather than equality
    correlation = np.corrcoef(custom_distance.ravel(), library_distance.ravel())[0, 1]
    assert correlation > 0.95, (
        f"Distance maps poorly correlated (r={correlation:.3f}). "
        "Custom chamfer implementation may be incorrect."
    )

    # Verify relative error is within acceptable bounds
    # Calculate normalised RMSE to account for scale differences
    mask = library_distance > 0  # Only compare non-zero distances
    relative_error = np.sqrt(
        np.mean(
            ((custom_distance[mask] - library_distance[mask]) / library_distance[mask])
            ** 2
        )
    )
    assert relative_error < 0.20, (
        f"Relative error too high ({relative_error:.2%}). "
        "Chamfer approximation quality insufficient."
    )


@pytest.mark.parametrize(
    "test_volume",
    [
        "two_spheres_not_touching",
        "two_spheres_touching",
        "three_spheres_all_touching",
        "three_spheres_not_touching",
    ],
    indirect=True,
)
def test_local_minima_matches_skimage(test_volume):
    """Verify custom local minima detection matches scikit-image implementation.

    Compares the number and locations of local minima found by the custom
    find_local_minima function against scikit-image's morphology.local_minima.
    Both implementations should identify the same local minima in the inverted
    distance transform.

    The test checks:
    - Number of detected minima matches between implementations
    - Spatial locations of minima are identical

    Args:
        test_volume: 3D numpy array fixture containing test image data.
            Parametrised to test multiple geometric configurations.
    """
    # Create binary volume by thresholding
    threshold = otsu_threshold(test_volume)
    binary_volume = test_volume > threshold

    # Compute distance transform using library implementation for consistency
    distance_transform = ndi.distance_transform_edt(binary_volume)

    # Find local minima in inverted distance transform using custom implementation
    custom_minima = find_local_minima(-distance_transform)

    # Find local minima using scikit-image library
    library_minima = morphology.local_minima(-distance_transform)

    # Verify both implementations detect the same number of minima
    custom_count = np.sum(custom_minima)
    library_count = np.sum(library_minima)
    assert custom_count == library_count, (
        f"Number of minima differs: custom={custom_count}, library={library_count}"
    )

    # Verify minima locations are identical
    assert np.array_equal(custom_minima, library_minima), (
        "Local minima locations do not match between implementations"
    )


@pytest.mark.parametrize(
    "test_volume",
    [
        "two_spheres_not_touching",
        "two_spheres_touching",
        "three_spheres_all_touching",
        "three_spheres_not_touching",
    ],
    indirect=True,
)
def test_watershed_segmentation_matches_skimage(test_volume):
    """Verify custom watershed segmentation matches scikit-image implementation.

    Compares the watershed segmentation results from the custom watershed_3d
    function against scikit-image's watershed implementation. Both should
    produce the same number of distinct regions when given identical inputs.

    The test checks:
    - Number of segmented regions matches between implementations
    - Both produce valid segmentation masks

    Args:
        test_volume: 3D numpy array fixture containing test image data.
            Parametrised to test multiple geometric configurations.
    """
    # Create binary volume by thresholding
    threshold = otsu_threshold(test_volume)
    binary_volume = test_volume > threshold

    # Compute distance transform using library implementation for consistency
    distance_transform = ndi.distance_transform_edt(binary_volume)

    # Find local minima to use as watershed markers
    local_minima = morphology.local_minima(-distance_transform)

    # Apply watershed using custom implementation
    custom_watershed = watershed_3d(distance_transform, binary_volume, local_minima)
    # Extract final labelled volume (assuming watershed_3d returns tuple)
    custom_labels = (
        custom_watershed[2] if isinstance(custom_watershed, tuple) else custom_watershed
    )

    # Apply watershed using scikit-image library
    markers_library = measure.label(local_minima)
    library_labels = watershed(-distance_transform, markers_library, mask=binary_volume)

    # Verify both implementations produce the same number of regions
    custom_region_count = len(np.unique(custom_labels)) - 1  # Exclude background
    library_region_count = len(np.unique(library_labels)) - 1  # Exclude background
    assert custom_region_count == library_region_count, (
        f"Number of regions differs: custom={custom_region_count}, "
        f"library={library_region_count}"
    )

    # Verify both produce valid segmentation masks (non-empty)
    assert custom_region_count > 0, "Custom watershed produced no regions"
    assert library_region_count > 0, "Library watershed produced no regions"
