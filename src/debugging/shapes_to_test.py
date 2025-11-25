"""Fixtures for test volumes with different sphere configurations."""

from shape_creation import create_three_spheres_example, create_two_spheres_example


def two_spheres_not_touching():
    """Test image with two isolated spheres."""
    return create_two_spheres_example(
        centre1=(6, 6, 6), centre2=(2, 2, 2), radius1=2, radius2=1
    )


def two_spheres_touching():
    """Test image with two spheres which are touching."""
    return create_two_spheres_example(
        centre1=(6, 4, 5), centre2=(4, 2, 2), radius1=2, radius2=2
    )


def three_spheres_all_touching():
    """Test image with three spheres all touching."""
    return create_three_spheres_example(
        centre1=(6, 4, 5),
        centre2=(4, 2, 2),
        centre3=(3, 7, 1),
        radius1=2,
        radius2=2,
        radius3=3,
    )


def three_spheres_not_touching():
    """Test image with three spheres none of which are touching."""
    return create_three_spheres_example(
        centre1=(7, 7, 7),
        centre2=(2, 2, 2),
        centre3=(4, 4, 4),
        radius1=2,
        radius2=1,
        radius3=1,
    )
