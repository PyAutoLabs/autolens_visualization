"""
Simulator: Multi-Galaxy Imaging
===============================

Simulates the imaging dataset of ``true_models.multi_galaxy`` — two lens galaxies with an
external shear, a light-only extra galaxy and a SersicCore source — and writes ``data.fits``,
``psf.fits``, ``noise_map.fits`` and ``tracer.json`` into ``dataset/multi_galaxy/simple/``.

Adapted from ``autolens_workspace/scripts/multi_galaxy/simulator.py``: same galaxies, PSF and
exposure, on a 140x140 (not 200x200) grid with a seeded noise realisation. The extra-galaxy
noise mask and the multiple-image positions are not written (the producer does not use them).

Usage
-----

    python scripts/misc/simulators/multi_galaxy.py
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

    from true_models import multi_galaxy as tm

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "multi_galaxy" / tm.DATASET_NAME
    dataset_path.mkdir(parents=True, exist_ok=True)

    grid = al.Grid2D.uniform(shape_native=tm.SHAPE_NATIVE, pixel_scales=tm.PIXEL_SCALE)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=tm.MAIN_LENS_CENTRES + [tm.EXTRA_GALAXY_CENTRE],
    )
    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11),
        sigma=0.08,
        pixel_scales=grid.pixel_scales,
        convolve_over_sample_size=1,
    )
    simulator = al.SimulatorImaging(
        exposure_time=900.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    tracer = al.Tracer(
        galaxies=[tm.lens_0(), tm.lens_1(), tm.extra_galaxy(), tm.source()],
        fields=[tm.field()],
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    aplt.fits_imaging(
        dataset=dataset,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )
    al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")
    print(f"  wrote {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    import argparse

    from autonerves import jax_wrapper  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instrument", default="simple", choices=["simple"])
    parser.add_argument("--output-root", type=Path, default=None)
    simulate(output_root=parser.parse_args().output_root)
