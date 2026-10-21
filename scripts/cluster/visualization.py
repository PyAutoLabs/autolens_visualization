"""
Visualization: Cluster
======================

Renders the figures ``VisualizerPoint`` writes during a cluster model-fit (two point sources at
z=1.0 and z=2.0 behind a multi-plane cluster lens), so the latest version of each lives in git
(``GALLERY.md``). A cluster model-fit has one ``AnalysisPoint`` per point-source dataset,
combined in a factor graph: each dataset's own figures are written to its own folder, and the
figures of every source together (where all sources are visible) to the top-level folder.

Output tree (wiped at the start of every run)::

    scripts/cluster/images/visualization/
        dataset.png                <- VisualizerPoint.visualize_before_fit_combined:
                                      "Dataset Positions" of every source (dataset_point.png)
        image_with_points.png      <- every source's positions over the cluster image
        fit.png                    <- VisualizerPoint.visualize_combined: "Fit Positions" of
                                      every source, each source's critical curves at its own z
        fit_per_image.png          <- one row per source: residual ranking (range shared by
                                      both sources) / image-plane zoom / source plane
        fit_image_with_points.png  <- every source's positions and curves over the image
        point_0/ point_1/          <- each dataset's own figures (visualize_before_fit +
                                      visualize): dataset.png, image_with_points.png, fit.png,
                                      fit_per_image.png, fit_image_with_points.png and
                                      point_solver.png (the model's real multi-plane PointSolver
                                      refining onto this source's images)

The per-dataset ``visualize`` also writes the tracer and galaxy figures, which this repo does
not track: they (and every other untracked file) are deleted, and ``dataset_point.png`` is
renamed ``dataset.png``.

Dataset: ``dataset/cluster/simple`` (``point_0``: 5 images, ``point_1``: 2 images, 5 mas noise,
and ``data.fits``: the cluster image, visualization only), from
``scripts/misc/simulators/cluster.py``. The model is that simulator's TRUE model
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
from autolens.point.model.visualizer import VisualizerPoint

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
__Cluster Image__

Visualization only: it is passed to every ``AnalysisPoint`` for the image-with-points figures and
never enters the point-source likelihood.
"""
image = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", hdu=0, pixel_scales=tm.IMAGE_PIXEL_SCALES
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
scratch_root.mkdir(parents=True)

"""
__Per Point-Source Dataset__
"""
analyses = []
for i in range(len(tm.SOURCE_REDSHIFTS)):
    dataset = al.from_json(file_path=dataset_path / f"point_dataset_{i}.json")
    analysis = al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        image=image,
        fit_positions_cls=al.FitPositionsSource,
        use_jax=False,
    )
    analyses.append(analysis)

    point_path = image_path / dataset.name
    point_path.mkdir(parents=True)
    scratch = scratch_root / dataset.name
    scratch.mkdir(parents=True)
    paths = SimpleNamespace(image_path=point_path, output_path=scratch)

    t0 = time.perf_counter()
    VisualizerPoint.visualize_before_fit(analysis=analysis, paths=paths, model=model)
    VisualizerPoint.visualize(
        analysis=analysis,
        paths=paths,
        instance=instance,
        during_analysis=False,
        quick_update=False,
    )
    keep_dataset_and_fit(
        point_path,
        dataset_png="dataset_point.png",
        extra_pngs=(
            "fit_per_image.png",
            "image_with_points.png",
            "fit_image_with_points.png",
            "point_solver.png",
        ),
    )
    print(f"visualize [{dataset.name}]: {time.perf_counter() - t0:.1f}s")

"""
__All Sources__

The figures of every source together, as ``VisualizerPoint`` writes them to the top-level image
folder of a factor-graph fit.
"""
t0 = time.perf_counter()
paths = SimpleNamespace(image_path=image_path, output_path=scratch_root)
VisualizerPoint.visualize_before_fit_combined(analyses=analyses, paths=paths, model=model)
VisualizerPoint.visualize_combined(
    analyses=analyses,
    paths=paths,
    instance=[instance] * len(analyses),
    during_analysis=False,
    quick_update=False,
)
keep_dataset_and_fit(
    image_path,
    dataset_png="dataset_point.png",
    extra_pngs=("fit_per_image.png", "image_with_points.png", "fit_image_with_points.png"),
    keep_dirs=tuple(analysis.dataset.name for analysis in analyses),
)
print(f"visualize [all sources]: {time.perf_counter() - t0:.1f}s")

n_png = len(list(image_path.rglob("*.png")))
print(f"Wrote {n_png} PNGs under {image_path.relative_to(REPO_ROOT)}")
