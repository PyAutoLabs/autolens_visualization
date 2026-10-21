"""Multi-galaxy: two lens galaxies, an external shear, a light-only extra galaxy and a source.

Adapted from ``autolens_workspace/scripts/multi_galaxy/simulator.py``; the image is cut down
from 200x200 to 140x140 pixels (7") so the tracked dataset stays small.
"""

import autolens as al

DATASET_NAME = "simple"
PIXEL_SCALE = 0.05
SHAPE_NATIVE = (140, 140)
MASK_RADIUS = 3.0
MAIN_LENS_CENTRES = [(0.35, 0.25), (-0.35, -0.25)]
EXTRA_GALAXY_CENTRE = (2.2, 1.6)


def lens_0():
    return al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.35, 0.25),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
            intensity=1.2,
            effective_radius=0.6,
            sersic_index=4.0,
        ),
        mass=al.mp.Isothermal(
            centre=(0.30, 0.28),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=45.0),
            einstein_radius=1.0,
        ),
    )


def lens_1():
    return al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(-0.35, -0.25),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
            intensity=1.0,
            effective_radius=0.5,
            sersic_index=4.0,
        ),
        mass=al.mp.Isothermal(
            centre=(-0.31, -0.22),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
            einstein_radius=0.8,
        ),
    )


def extra_galaxy():
    return al.Galaxy(
        redshift=0.5,
        light=al.lp.ExponentialSph(centre=EXTRA_GALAXY_CENTRE, intensity=1.0, effective_radius=0.3),
    )


def source():
    return al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.03),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=3.0,
            effective_radius=0.15,
            sersic_index=1.0,
        ),
    )


def field():
    return al.MassField(redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05))
