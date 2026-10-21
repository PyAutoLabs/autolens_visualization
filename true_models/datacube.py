"""Interferometer datacube: an isothermal lens + shear and a source whose brightness and centre
vary across a few spectral channels.

Adapted from ``autolens_workspace/scripts/interferometer/features/datacube/simulator.py``, cut
down to three channels and observed with this repo's ``sma`` preset (its uv-coverage and
``noise_sigma``) so the dirty images show the ring.
"""

import math

import autolens as al

DATASET_NAME = "datacube"
N_CHANNELS = 3
PEAK_CHANNEL = 1.0
SIGMA_CHANNEL = 1.2
PEAK_INTENSITY = 0.6
CENTRE_SHIFT_TOTAL = 0.12
SOURCE_CENTRE_BASE = (0.1, 0.1)


def channel_intensity(channel: int) -> float:
    return PEAK_INTENSITY * math.exp(-0.5 * ((channel - PEAK_CHANNEL) / SIGMA_CHANNEL) ** 2)


def channel_centre(channel: int) -> tuple:
    """The source centre shifts linearly in y across the channels (0.12" in total)."""
    fractional = (channel - (N_CHANNELS - 1) / 2.0) / (N_CHANNELS - 1)
    return (SOURCE_CENTRE_BASE[0] + CENTRE_SHIFT_TOTAL * fractional, SOURCE_CENTRE_BASE[1])


def lens():
    return al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )


def field():
    return al.MassField(redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05))


def source(channel: int):
    return al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=channel_centre(channel),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=channel_intensity(channel),
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )
