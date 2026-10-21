"""Weak lensing: a cluster-scale isothermal lens and a catalogue of background shears.

Adapted from ``autolens_workspace/scripts/weak/simulator.py`` (1500 sources in a 50"-200"
annulus, shape noise 0.25).
"""

import autolens as al

DATASET_NAME = "simple"
N_GALAXIES = 1500
RADIUS_INNER = 50.0
RADIUS_OUTER = 200.0
NOISE_SIGMA = 0.25
SEED = 1


def lens():
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=25.0,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
        ),
    )


def source():
    return al.Galaxy(redshift=1.0)
