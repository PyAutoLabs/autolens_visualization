"""
Simulator: Multi-Dataset Imaging
================================

Simulates the two-band imaging dataset of ``true_models.multi_dataset`` — one lens observed
in the g and r bands at different resolutions — and writes ``<band>_data.fits``,
``<band>_psf.fits``, ``<band>_noise_map.fits`` and ``<band>_tracer.json`` for each band into
``dataset/multi_dataset/lens_sersic/``.

Adapted from ``autolens_workspace/scripts/multi_dataset/simulator.py``: same galaxies, PSFs and
sky levels, on 80x80 (not 150x150) grids with seeded noise. The extra-galaxy noise masks are
not written (the producer does not use them).

Usage
-----

    python scripts/misc/simulators/multi_dataset.py
"""

import sys as _sys
from pathlib import Path as _Path


def _repo_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_visualization root (ruff.toml) not found")


if str(_repo_root()) not in _sys.path:
    _sys.path.insert(0, str(_repo_root()))

from pathlib import Path

_REPO_ROOT = _repo_root()


def simulate(output_root: Path | None = None) -> Path:
    import autolens as al
    import autolens.plot as aplt

    from true_models import multi_dataset as tm

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "multi_dataset" / tm.DATASET_NAME
    dataset_path.mkdir(parents=True, exist_ok=True)

    for band in tm.BANDS:  # g first: the hook's sentinel is the last band's r_data.fits
        grid = al.Grid2D.uniform(shape_native=tm.SHAPE_NATIVE, pixel_scales=tm.PIXEL_SCALES[band])
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0), tm.EXTRA_GALAXY_CENTRE],
        )
        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

        psf = al.Convolver.from_gaussian(
            convolve_over_sample_size=1,
            shape_native=(11, 11),
            sigma=tm.PSF_SIGMA[band],
            pixel_scales=grid.pixel_scales,
        )
        simulator = al.SimulatorImaging(
            exposure_time=300.0,
            psf=psf,
            background_sky_level=tm.BACKGROUND_SKY[band],
            add_poisson_noise_to_data=True,
            noise_seed=1,
        )
        tracer = al.Tracer(
            galaxies=[tm.lens(band), tm.extra_galaxy(band), tm.source(band)],
            fields=[tm.field()],
        )
        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

        aplt.fits_imaging(
            dataset=dataset,
            data_path=dataset_path / f"{band}_data.fits",
            psf_path=dataset_path / f"{band}_psf.fits",
            noise_map_path=dataset_path / f"{band}_noise_map.fits",
            overwrite=True,
        )
        al.output_to_json(obj=tracer, file_path=dataset_path / f"{band}_tracer.json")

    print(f"  wrote {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    import argparse

    from autonerves import jax_wrapper  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instrument", default="lens_sersic", choices=["lens_sersic"])
    parser.add_argument("--output-root", type=Path, default=None)
    simulate(output_root=parser.parse_args().output_root)
