"""
Simulator: Weak Lensing
=======================

Simulates the weak-lensing shear catalogue of ``true_models.weak`` — 1500 background
galaxies in a 50"-200" annulus around a cluster-scale isothermal lens, with shape noise 0.25 —
and writes ``dataset.json`` (the ``WeakDataset``) and ``tracer.json`` into
``dataset/weak/simple/``.

Adapted from ``autolens_workspace/scripts/weak/simulator.py`` (same lens, catalogue geometry,
noise and seeds); the PNG previews are not written.

Usage
-----

    python scripts/misc/simulators/weak.py
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
    import numpy as np

    from true_models import weak as tm

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "weak" / tm.DATASET_NAME
    dataset_path.mkdir(parents=True, exist_ok=True)

    tracer = al.Tracer(galaxies=[tm.lens(), tm.source()])

    rng = np.random.default_rng(tm.SEED)
    radii = np.sqrt(rng.uniform(tm.RADIUS_INNER**2.0, tm.RADIUS_OUTER**2.0, tm.N_GALAXIES))
    phi = rng.uniform(0.0, 2.0 * np.pi, tm.N_GALAXIES)
    positions = al.Grid2DIrregular(
        values=np.stack([radii * np.sin(phi), radii * np.cos(phi)], axis=1)
    )

    simulator = al.SimulatorShearYX(noise_sigma=tm.NOISE_SIGMA, seed=tm.SEED)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=positions, name=tm.DATASET_NAME)

    al.output_to_json(obj=dataset, file_path=dataset_path / "dataset.json")
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
