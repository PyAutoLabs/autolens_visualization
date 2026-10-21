"""
Simulator: Cluster (Multi-Plane Point Sources)
==============================================

Simulates the point-source datasets of ``true_models.cluster`` — the multiple images of two
point sources at z=1.0 (``point_0``) and z=2.0 (``point_1``) behind a cluster lens (BCG,
satellite, ten scaling-relation members and an NFW host halo) — and writes
``point_dataset_0.json``, ``point_dataset_1.json`` and ``tracer.json`` into
``dataset/cluster/simple/``.

Adapted from ``autolens_workspace/scripts/cluster/simulator.py``: same galaxies and 5 mas
position noise (the workspace adds none; seeded here), solved with the NumPy ``PointSolver`` on
a coarser 400x400 @ 0.2" grid instead of the JIT-compiled 800x800 @ 0.1" one. The workspace's
1000x1000 imaging, CSV tables and PNG previews are not written (the producer fits positions).

Usage
-----

    python scripts/misc/simulators/cluster.py
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

    from true_models import cluster as tm

    root = output_root if output_root is not None else _REPO_ROOT
    dataset_path = root / "dataset" / "cluster" / tm.DATASET_NAME
    dataset_path.mkdir(parents=True, exist_ok=True)

    tracer = al.Tracer(
        galaxies=tm.main_lenses() + tm.scaling_galaxies() + [tm.host_halo()] + tm.sources()
    )
    solver = al.PointSolver.for_grid(
        grid=tm.solver_grid(), pixel_scale_precision=0.001, magnification_threshold=0.1
    )

    rng = np.random.default_rng(1)
    for i, (centre, z) in enumerate(zip(tm.SOURCE_CENTRES, tm.SOURCE_REDSHIFTS)):
        positions = np.asarray(
            solver.solve(tracer=tracer, source_plane_coordinate=centre, plane_redshift=z).array
        )
        positions = positions[np.isfinite(positions).all(axis=1)]
        positions = positions + rng.normal(0.0, tm.POSITION_NOISE, size=positions.shape)
        dataset = al.PointDataset(
            name=f"point_{i}",
            positions=al.Grid2DIrregular(values=positions),
            positions_noise_map=tm.POSITION_NOISE,
            redshift=z,
        )
        al.output_to_json(obj=dataset, file_path=dataset_path / f"point_dataset_{i}.json")
        print(f"  point_{i} (z={z}): {len(positions)} images")

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
