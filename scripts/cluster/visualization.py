"""
Visualization: Cluster
======================

Renders the two figures ``VisualizerPoint`` writes for each point-source dataset of a
multi-plane cluster fit (two point sources at z=1.0 and z=2.0 behind a cluster lens), so the
latest version of each lives in git (``GALLERY.md``). A cluster model-fit has one
``AnalysisPoint`` per point-source dataset (combined in a factor graph), each visualized
separately, so this producer calls the visualizer once per dataset. The figures are the
generic point-source plots — bespoke cluster figures are a follow-up.

Output tree (wiped at the start of every run)::

    scripts/cluster/images/visualization/
        point_0/dataset.png, point_0/fit.png   <- PlotterPoint.dataset_point, PlotterPoint.fit_point
        point_1/dataset.png, point_1/fit.png

**Plotter, not Visualizer.** ``VisualizerPoint.visualize`` also renders the tracer and galaxy
figures this repo does not track, so this producer calls the same two ``PlotterPoint`` methods the
visualizer calls (``dataset_point(solver=...)`` in ``visualize_before_fit``;
``fit_point(quick_update=True)`` in ``visualize``), writing the figures the visualizer would. The
"Fit Positions" panel draws the tracer's critical curves, which ``fit_point`` computes on a 100x100
grid covering the solver grid: for this 15-galaxy multi-plane lens that costs ~2 min per dataset.

The point plotter writes ``dataset_point.png`` (renamed to ``dataset.png``); every other file
the visualizer writes is deleted.

Dataset: ``dataset/cluster/simple`` (``point_0``: 5 images, ``point_1``: 2 images, 5 mas noise),
from ``scripts/misc/simulators/cluster.py``. The model is that simulator's TRUE model
(``true_models/cluster.py``), fitted with the source-plane chi-squared
(``FitPositionsSource``) the workspace cluster example uses, with free-centre point sources.

Run from the repo root::

    python scripts/cluster/visualization.py
"""

import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    for _p in Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_visualization root (ruff.toml) not found")


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

# Push the all-true plots.yaml before any visualization code path reads config.
from autolens import conf

conf.instance.push(
    new_path=str(REPO_ROOT / "config"),
    output_path=str(REPO_ROOT / "scripts" / "cluster" / "images"),
)

import autofit as af
import autolens as al
from autolens.point.model.plotter import PlotterPoint

from _viz_cli import ensure_dataset, keep_dataset_and_fit
from true_models import cluster as tm

STEM = Path(__file__).stem

dataset_path = ensure_dataset("cluster", workspace_root=REPO_ROOT)

"""
__True Model__

The simulator's galaxies, every parameter fixed: the prior-median instance IS the true model.
Lens galaxies, scaling members and halo go in ``galaxies`` alongside the two sources.
"""
model = af.Collection(
    galaxies=af.Collection(
        *tm.main_lenses(), *tm.scaling_galaxies(), tm.host_halo(), *tm.sources()
    ),
)
instance = model.instance_from_prior_medians()

solver = al.PointSolver.for_grid(
    grid=tm.solver_grid(), pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "cluster" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch_root = REPO_ROOT / "output" / "visualization" / "cluster" / STEM
if scratch_root.exists():
    shutil.rmtree(scratch_root)

"""
__Per Point-Source Dataset__
"""
for i in range(len(tm.SOURCE_REDSHIFTS)):
    dataset = al.from_json(file_path=dataset_path / f"point_dataset_{i}.json")
    analysis = al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        fit_positions_cls=al.FitPositionsSource,
        use_jax=False,
    )

    point_path = image_path / dataset.name
    point_path.mkdir(parents=True)
    scratch = scratch_root / dataset.name
    scratch.mkdir(parents=True)
    paths = SimpleNamespace(image_path=point_path, output_path=scratch)

    t0 = time.perf_counter()
    plotter = PlotterPoint(image_path=paths.image_path, title_prefix=analysis.title_prefix)
    plotter.dataset_point(dataset=analysis.dataset, solver=analysis.solver)
    plotter.fit_point(
        fit=analysis.fit_for_visualization(instance=instance),
        quick_update=True,
    )
    keep_dataset_and_fit(point_path, dataset_png="dataset_point.png")
    print(f"visualize [{dataset.name}]: {time.perf_counter() - t0:.1f}s")

n_png = len(list(image_path.rglob("*.png")))
print(f"Wrote {n_png} PNGs under {image_path.relative_to(REPO_ROOT)}")
