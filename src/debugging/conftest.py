"""Fixtures for test volumes with different sphere configurations."""

import pytest


@pytest.fixture
def two_spheres_not_touching():
    """Test image with two isolated spheres.

    Returns:
        ndarray: 3D volume containing two non-touching spheres.
    """
    from debugging.shapes_to_test import two_spheres_not_touching

    return two_spheres_not_touching()


@pytest.fixture
def two_spheres_touching():
    """Test image with two spheres which are touching.

    Returns:
        ndarray: 3D volume containing two touching spheres.
    """
    from debugging.shapes_to_test import two_spheres_touching

    return two_spheres_touching()


@pytest.fixture
def three_spheres_all_touching():
    """Test image with three spheres all touching.

    Returns:
        ndarray: 3D volume containing three touching spheres.
    """
    from debugging.shapes_to_test import three_spheres_all_touching

    return three_spheres_all_touching()


@pytest.fixture
def three_spheres_not_touching():
    """Test image with three spheres none of which are touching.

    Returns:
        ndarray: 3D volume containing three non-touching spheres.
    """
    from debugging.shapes_to_test import three_spheres_not_touching

    return three_spheres_not_touching()


@pytest.fixture
def test_volume(request):
    """Indirect fixture to parametrise tests with different volume fixtures.

    Enables pytest.mark.parametrize to accept fixture names as strings
    and dynamically load the corresponding fixtures.

    Args:
        request: pytest request object containing parametrised fixture name.

    Returns:
        ndarray: The requested test volume fixture.
    """
    return request.getfixturevalue(request.param)
