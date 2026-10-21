"""Cluster: a multi-plane cluster lens with two point sources at different redshifts.

Adapted from ``autolens_workspace/scripts/cluster/simulator.py``: a BCG and a satellite (dPIE
mass + Sersic light), ten scaling-relation members, an NFW host halo at z=0.5, and two point
sources at z=1.0 and z=2.0 (``point_0`` / ``point_1``), so ray-tracing is multi-plane. Only the
point-source positions are simulated (the workspace's 1000x1000 imaging is not needed here).
"""

import autolens as al

DATASET_NAME = "simple"
REDSHIFT_LENS = 0.5
SOURCE_REDSHIFTS = [1.0, 2.0]
SOURCE_CENTRES = [(0.3, 0.5), (-0.8, 1.2)]
POSITION_NOISE = 0.005

MAIN_LENS_CENTRES = [(0.0, 0.0), (10.0, 8.0)]
MAIN_LENS_DPIE = [(0.0, 20.0, 330.0), (0.0, 12.0, 210.0)]  # (r_core, r_cut, sigma)
MAIN_LENS_SERSIC = [(1.5, 3.0, 4.0), (0.8, 1.5, 3.5)]  # (intensity, effective_radius, n)

SCALING_CENTRES = [
    (5.5, -6.5),
    (-7.5, 3.0),
    (12.0, -5.0),
    (-4.0, -9.0),
    (3.0, 13.0),
    (-14.0, 4.0),
    (15.0, 9.0),
    (-9.0, -12.0),
    (8.5, 5.5),
    (-6.5, 11.0),
]
SCALING_LUMINOSITIES = [0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05]
SCALING_SIGMA_REF = 85.0
SCALING_SIGMA_EXPONENT = 0.25
SCALING_GAMMA = 0.2
SCALING_RCUT_EXPONENT = 1.0 + SCALING_GAMMA - 2.0 * SCALING_SIGMA_EXPONENT
SCALING_R_CUT_REF = 5.0


def main_lenses():
    galaxies = []
    for centre, (r_core, r_cut, sigma), (intensity, r_eff, n) in zip(
        MAIN_LENS_CENTRES, MAIN_LENS_DPIE, MAIN_LENS_SERSIC
    ):
        galaxies.append(
            al.Galaxy(
                redshift=REDSHIFT_LENS,
                bulge=al.lp.SersicSph(
                    centre=centre, intensity=intensity, effective_radius=r_eff, sersic_index=n
                ),
                mass=al.mp.dPIEMassSph(
                    centre=centre,
                    sigma=sigma,
                    r_core=r_core,
                    r_cut=r_cut,
                    redshift_object=REDSHIFT_LENS,
                    redshift_source=max(SOURCE_REDSHIFTS),
                ),
            )
        )
    return galaxies


def scaling_galaxies():
    galaxies = []
    for centre, luminosity in zip(SCALING_CENTRES, SCALING_LUMINOSITIES):
        galaxies.append(
            al.Galaxy(
                redshift=REDSHIFT_LENS,
                bulge=al.lp.SersicSph(
                    centre=centre, intensity=luminosity, effective_radius=0.8, sersic_index=3.0
                ),
                mass=al.mp.dPIEMassSph(
                    centre=centre,
                    sigma=SCALING_SIGMA_REF * luminosity**SCALING_SIGMA_EXPONENT,
                    r_core=0.0,
                    r_cut=SCALING_R_CUT_REF * luminosity**SCALING_RCUT_EXPONENT,
                    redshift_object=REDSHIFT_LENS,
                    redshift_source=max(SOURCE_REDSHIFTS),
                ),
            )
        )
    return galaxies


def host_halo():
    return al.Galaxy(
        redshift=REDSHIFT_LENS,
        dark=al.mp.NFWMCRLudlowSph(
            centre=(0.0, 0.0),
            mass_at_200=10**15.3,
            redshift_object=REDSHIFT_LENS,
            redshift_source=max(SOURCE_REDSHIFTS),
        ),
    )


def sources():
    galaxies = []
    for i, (centre, z) in enumerate(zip(SOURCE_CENTRES, SOURCE_REDSHIFTS)):
        galaxies.append(
            al.Galaxy(
                redshift=z,
                bulge=al.lp.SersicCore(
                    centre=centre,
                    ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
                    intensity=2.0,
                    effective_radius=0.3,
                    sersic_index=1.0,
                ),
                **{f"point_{i}": al.ps.Point(centre=centre)},
            )
        )
    return galaxies


def solver_grid():
    """Coarser than the workspace's JAX 800x800 @ 0.1" grid: the NumPy solve stays quick."""
    return al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=0.2)
