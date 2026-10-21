"""Point source: an isothermal lens with one point source behind it.

Adapted from ``autolens_workspace/scripts/point_source/simulator.py``.
"""

import autolens as al

DATASET_NAME = "simple"
POINT_NAME = "point_0"
POSITION_NOISE = 0.005  # arcsec, HST-like point-source astrometry
SOURCE_CENTRE = (0.07, 0.07)


def lens():
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )


def source():
    return al.Galaxy(
        redshift=1.0,
        light=al.lp.ExponentialCore(
            centre=SOURCE_CENTRE, intensity=0.1, effective_radius=0.02, radius_break=0.025
        ),
        point_0=al.ps.Point(centre=SOURCE_CENTRE),
    )


def solver_grid():
    """The image-plane grid the ``PointSolver`` starts from (simulator and producer)."""
    return al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)
