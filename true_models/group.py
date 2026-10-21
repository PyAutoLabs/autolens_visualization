"""Group: one main lens, two extra galaxies with dPIE mass, and a source.

Adapted from ``autolens_workspace/scripts/group/simulator.py``; the image is cut down from
250x250 to 160x160 pixels (16") — still enclosing the 7.5" mask and both extra galaxies — so
the tracked dataset stays small.
"""

import autolens as al

DATASET_NAME = "simple"
PIXEL_SCALE = 0.1
SHAPE_NATIVE = (160, 160)
MASK_RADIUS = 7.5
MAIN_LENS_CENTRES = [(0.0, 0.0)]
EXTRA_GALAXIES_CENTRES = [(3.5, 2.5), (-4.4, -5.0)]


def lens_0():
    return al.Galaxy(
        redshift=0.5,
        bulge=al.lp.SersicSph(
            centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
        ),
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
    )


def _extra_galaxy(centre, sigma):
    return al.Galaxy(
        redshift=0.5,
        bulge=al.lp.SersicSph(centre=centre, intensity=0.9, effective_radius=0.8, sersic_index=3.0),
        mass=al.mp.dPIEMassSph(
            centre=centre,
            sigma=sigma,
            r_core=0.0,
            r_cut=10.0,
            redshift_object=0.5,
            redshift_source=1.0,
        ),
    )


def extra_galaxies():
    return [
        _extra_galaxy(EXTRA_GALAXIES_CENTRES[0], sigma=212.0),
        _extra_galaxy(EXTRA_GALAXIES_CENTRES[1], sigma=239.0),
    ]


def source():
    return al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.1),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=3.0,
            effective_radius=0.4,
            sersic_index=1.0,
        ),
    )
