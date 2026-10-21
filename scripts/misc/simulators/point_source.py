"""
Simulator: Point Source
=======================

Simulates the point-source dataset of ``true_models.point_source`` — the image-plane positions
of one point source (``point_0``) behind an isothermal lens — and writes
``point_dataset.json`` (the ``PointDataset``, positions only) and ``tracer.json`` into
``dataset/point_source/simple/``, plus the accompanying imaging (``data.fits``, ``psf.fits``,
``noise_map.fits``) the producers plot the positions over (``image_with_points.png``).

The imaging places one unlensed ``al.lp.PointSource`` per noisy image position on an
image-plane galaxy at the lens redshift (a delta function is not lensed: its multiple images
come from the ``PointSolver``), with flux ``SOURCE_FLUX`` x |magnification|, convolved with a
normalized Gaussian PSF.

Adapted from ``autolens_workspace/scripts/point_source/simulator.py``: same lens, source,
5 mas position noise and imaging recipe, but all noise is seeded so the dataset regenerates
identically, and the flux / time-delay variants are not written (the producers fit positions
only).

Usage
-----

    python scripts/misc/simulators/point_source.py
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
    import numpy as np

    from true_models import point_source as tm

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "point_source" / tm.DATASET_NAME
    dataset_path.mkdir(parents=True, exist_ok=True)

    lens, source = tm.lens(), tm.source()
    tracer = al.Tracer(galaxies=[lens, source])

    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05),
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )
    positions = solver.solve(tracer=tracer, source_plane_coordinate=tm.SOURCE_CENTRE)

    rng = np.random.default_rng(1)
    positions = np.asarray(positions.array)
    positions_with_noise = positions + rng.normal(0.0, tm.POSITION_NOISE, size=positions.shape)

    dataset = al.PointDataset(
        name=tm.POINT_NAME,
        positions=al.Grid2DIrregular(values=positions_with_noise),
        positions_noise_map=tm.POSITION_NOISE,
    )

    al.output_to_json(obj=dataset, file_path=dataset_path / "point_dataset.json")
    al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")

    magnifications = al.LensCalc.from_tracer(tracer=tracer).magnification_2d_via_hessian_from(
        grid=al.Grid2DIrregular(values=positions_with_noise)
    )
    image_plane_galaxy = al.Galaxy(
        redshift=lens.redshift,
        **{
            f"image_{i}": al.lp.PointSource(
                centre=tuple(position), intensity=tm.SOURCE_FLUX * float(np.abs(magnification))
            )
            for i, (position, magnification) in enumerate(zip(positions_with_noise, magnifications))
        },
    )
    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=tm.IMAGE_PIXEL_SCALES)
    psf = al.Convolver.from_gaussian(
        convolve_over_sample_size=1,
        shape_native=(11, 11),
        sigma=0.1,
        pixel_scales=grid.pixel_scales,
        normalize=True,
    )
    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )
    imaging = simulator.via_tracer_from(tracer=al.Tracer(galaxies=[image_plane_galaxy]), grid=grid)
    aplt.fits_imaging(
        dataset=imaging,
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        overwrite=True,
    )
    print(f"  wrote {dataset_path} ({len(positions)} images)")
    return dataset_path


if __name__ == "__main__":
    import argparse

    from autonerves import jax_wrapper  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instrument", default="simple", choices=["simple"])
    parser.add_argument("--output-root", type=Path, default=None)
    simulate(output_root=parser.parse_args().output_root)
