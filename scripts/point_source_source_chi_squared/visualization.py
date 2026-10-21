"""
Visualization: Point Source (Source-Plane Chi-Squared)
======================================================

Renders the two figures ``VisualizerPoint`` writes for a point-source fit that uses the
source-plane chi-squared (``FitPositionsSource``: the observed positions are ray-traced to the
source plane and compared with the point source's centre, magnification-weighted), so the
latest version of each lives in git (``GALLERY.md``).

Output tree (wiped at the start of every run)::

    scripts/point_source_source_chi_squared/images/visualization/
        dataset.png   <- visualize_before_fit (the plotter writes dataset_point.png; renamed)
        fit.png       <- visualize(quick_update=True)
        image_with_points.png <- visualize(quick_update=True): the observed positions and the
                                 model's positions + critical curves over ``data.fits``

Every other file the visualizer writes is deleted: this domain tracks only these three
figures. ``image_with_points.png`` is written before and after the fit to the same file;
the kept one is the after-fit version, with the model positions.

Dataset: ``dataset/point_source/simple`` (four image-plane positions, 5 mas noise; shared by
both point-source chi-squared producers), from
``scripts/misc/simulators/point_source.py``. The model is that simulator's TRUE model
(``true_models/point_source.py``) with a free-centre ``al.ps.Point``.

Run from the repo root::

    python scripts/point_source_source_chi_squared/visualization.py
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
    output_path=str(REPO_ROOT / "scripts" / "point_source_source_chi_squared" / "images"),
)

import autofit as af
import autolens as al
from autolens.point.model.visualizer import VisualizerPoint

from _viz_cli import ensure_dataset, keep_dataset_and_fit
from true_models import point_source as tm

STEM = Path(__file__).stem

"""
__Dataset__
"""
dataset_path = ensure_dataset("point_source", workspace_root=REPO_ROOT)
dataset = al.from_json(file_path=dataset_path / "point_dataset.json")

"""
The accompanying imaging, used only for visualization (``image_with_points.png``), not in the
likelihood.
"""
data = al.Array2D.from_fits(
    file_path=dataset_path / "data.fits", hdu=0, pixel_scales=tm.IMAGE_PIXEL_SCALES
)

"""
__True Model__

The simulator's lens and source, every parameter fixed: the prior-median instance IS the true
model.
"""
model = af.Collection(galaxies=af.Collection(lens=tm.lens(), source=tm.source()))

"""
__Analysis__

``FitPositionsSource`` never solves the lens equation, but ``AnalysisPoint`` always takes a
``PointSolver`` (the same one as ``point_source_image_chi_squared``).
"""
solver = al.PointSolver.for_grid(
    grid=tm.solver_grid(), pixel_scale_precision=0.001, magnification_threshold=0.1
)

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    image=data,
    fit_positions_cls=al.FitPositionsSource,
    use_jax=False,
)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "point_source_source_chi_squared" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch = REPO_ROOT / "output" / "visualization" / "point_source_source_chi_squared" / STEM
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)

paths = SimpleNamespace(image_path=image_path, output_path=scratch)

"""
__Visualize__
"""
t0 = time.perf_counter()
VisualizerPoint.visualize_before_fit(analysis=analysis, paths=paths, model=model)
VisualizerPoint.visualize(
    analysis=analysis,
    paths=paths,
    instance=model.instance_from_prior_medians(),
    during_analysis=False,
    quick_update=True,
)
keep_dataset_and_fit(
    image_path, dataset_png="dataset_point.png", extra_pngs=("image_with_points.png",)
)
print(f"visualize: {time.perf_counter() - t0:.1f}s")
print(
    f"Wrote {sorted(p.name for p in image_path.iterdir())} under {image_path.relative_to(REPO_ROOT)}"
)
