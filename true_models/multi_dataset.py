"""Multi-dataset: the same lens observed in two wavebands (g, r) at different resolutions.

Adapted from ``autolens_workspace/scripts/multi_dataset/simulator.py``: one shared isothermal
mass + shear, per-band Sersic lens light, SersicCore source and light-only extra galaxy. The
images are cut down from 150x150 to 80x80 pixels so the tracked dataset stays small.
"""

import autolens as al

DATASET_NAME = "lens_sersic"
BANDS = ["g", "r"]
PIXEL_SCALES = {"g": 0.08, "r": 0.12}
PSF_SIGMA = {"g": 0.1, "r": 0.2}
BACKGROUND_SKY = {"g": 0.1, "r": 0.15}
SHAPE_NATIVE = (80, 80)
MASK_RADIUS = 3.0
EXTRA_GALAXY_CENTRE = (2.2, 1.6)

_LENS_INTENSITY = {"g": 0.05, "r": 1.5}
_SOURCE_INTENSITY = {"g": 0.5, "r": 0.7}
_EXTRA_INTENSITY = {"g": 0.4, "r": 1.0}


def lens(band):
    return al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=_LENS_INTENSITY[band],
            effective_radius=0.8,
            sersic_index=4.0,
        ),
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )


def extra_galaxy(band):
    return al.Galaxy(
        redshift=0.5,
        light=al.lp.ExponentialSph(
            centre=EXTRA_GALAXY_CENTRE, intensity=_EXTRA_INTENSITY[band], effective_radius=0.3
        ),
    )


def source(band):
    return al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=_SOURCE_INTENSITY[band],
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )


def field():
    return al.MassField(redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05))
