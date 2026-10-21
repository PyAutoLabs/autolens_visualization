"""
Simulator: Interferometer Datacube
==================================

Simulates the interferometer datacube of ``true_models.datacube`` — three spectral channels of
one lens whose source brightness and centre vary with channel — and writes, per channel,
``channel_<i>/{data,noise_map,uv_wavelengths}.fits`` + ``tracer.json`` into
``dataset/interferometer/datacube/``.

Adapted from ``autolens_workspace/scripts/interferometer/features/datacube/simulator.py``: the
same lens, shear and channel-dependent source, cut down to three channels, observed with this
repo's ``sma`` preset (the uv-coverage of ``dataset/interferometer/sma``, its real-space grid
and ``noise_sigma``) instead of the workspace's ``noise_sigma=1000``, so the dirty images show
the ring. The stacked 3D / 4D FITS cubes are not written (the producer fits channel by channel).

Usage
-----

    python scripts/misc/simulators/datacube.py
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

    from instruments.interferometer import INSTRUMENTS
    from true_models import datacube as tm

    config = INSTRUMENTS["sma"]
    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "interferometer" / tm.DATASET_NAME
    dataset_path.mkdir(parents=True, exist_ok=True)

    uv_wavelengths = al.ndarray_via_fits_from(
        file_path=root / "dataset" / "interferometer" / "sma" / "uv_wavelengths.fits", hdu=0
    )
    grid = al.Grid2D.uniform(
        shape_native=config["real_space_shape"], pixel_scales=config["pixel_scale"]
    )

    for channel in range(tm.N_CHANNELS):
        channel_path = dataset_path / f"channel_{channel:03d}"
        channel_path.mkdir(parents=True, exist_ok=True)

        tracer = al.Tracer(galaxies=[tm.lens(), tm.source(channel)], fields=[tm.field()])
        simulator = al.SimulatorInterferometer(
            uv_wavelengths=uv_wavelengths,
            exposure_time=300.0,
            noise_sigma=config["noise_sigma"],
            transformer_class=al.TransformerDFT,
            noise_seed=config["seed"] + channel,
        )
        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

        aplt.fits_interferometer(
            dataset=dataset,
            data_path=channel_path / "data.fits",
            noise_map_path=channel_path / "noise_map.fits",
            uv_wavelengths_path=channel_path / "uv_wavelengths.fits",
            overwrite=True,
        )
        al.output_to_json(obj=tracer, file_path=channel_path / "tracer.json")
        print(f"  channel {channel:03d}: intensity={tm.channel_intensity(channel):.4f}")

    print(f"  wrote {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    import argparse

    from autonerves import jax_wrapper  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instrument", default="datacube", choices=["datacube"])
    parser.add_argument("--output-root", type=Path, default=None)
    simulate(output_root=parser.parse_args().output_root)
