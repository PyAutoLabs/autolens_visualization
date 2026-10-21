"""
Visualization: Point Source (Image-Plane Chi-Squared)
=====================================================

Renders the figures ``VisualizerPoint`` writes for a point-source fit that uses the
image-plane chi-squared (``FitPositionsImagePair``: the ``PointSolver`` finds the model's
multiple images, which are paired one-to-one with the observed positions), so the latest
version of each lives in git (``GALLERY.md``), plus a coarse-grid teaching version of the
``PointSolver`` refinement figure and its animation.

Output tree (wiped at the start of every run)::

    scripts/point_source_image_chi_squared/images/visualization/
        dataset.png   <- visualize_before_fit (the plotter writes dataset_point.png; renamed)
        fit.png       <- visualize: Fit Positions + Fit Source Plane (traced positions for
                         context only: the image-plane chi-squared has no source-plane noise)
        image_with_points.png     <- visualize_before_fit: the observed positions over
                                     ``data.fits`` (dataset only, no model)
        fit_image_with_points.png <- visualize: the observed positions, the model's
                                     positions and critical curves over ``data.fits``
        point_solver.png          <- visualize (``fit_point_dataset: subplot_point_solver``,
                                     on in this repo's plots.yaml, off in the library): the
                                     model's real ``PointSolver`` refining onto the images,
                                     steps 0, 1, 2 and final
        teaching/point_solver.png <- ``aplt.plot_point_solver`` with a coarse teaching grid
                                     (11x11 at 0.5", precision 0.01"), so the triangles stay
                                     visible at every step
        teaching/point_solver.gif <- the same, ``animate=True``: one step per frame, looping
                                     (not in the PNG manifest; tracked beside the still)

Every other file the visualizer writes is deleted: this domain tracks only these figures.

Dataset: ``dataset/point_source/simple`` (four image-plane positions, 5 mas noise; shared by
both point-source chi-squared producers), from
``scripts/misc/simulators/point_source.py``. The model is that simulator's TRUE model
(``true_models/point_source.py``) with a free-centre ``al.ps.Point``.

Run from the repo root::

    python scripts/point_source_image_chi_squared/visualization.py
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
    output_path=str(REPO_ROOT / "scripts" / "point_source_image_chi_squared" / "images"),
)

import autofit as af
import autolens as al
import autolens.plot as aplt
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
The accompanying imaging, used only for visualization (``image_with_points.png``,
``fit_image_with_points.png``), not in the likelihood.
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

``FitPositionsImagePair`` needs the ``PointSolver`` to locate the model's multiple images.
"""
solver = al.PointSolver.for_grid(
    grid=tm.solver_grid(), pixel_scale_precision=0.001, magnification_threshold=0.1
)

analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    image=data,
    fit_positions_cls=al.FitPositionsImagePair,
    use_jax=False,
)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "point_source_image_chi_squared" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch = REPO_ROOT / "output" / "visualization" / "point_source_image_chi_squared" / STEM
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
    quick_update=False,
)
keep_dataset_and_fit(
    image_path,
    dataset_png="dataset_point.png",
    extra_pngs=(
        "image_with_points.png",
        "fit_image_with_points.png",
        "point_solver.png",
    ),
)
print(f"visualize: {time.perf_counter() - t0:.1f}s")

"""
__Teaching Figure__

The model's real solver grid (0.05" triangles) makes every step after step 0 sub-pixel, so a
coarse grid (11x11 at 0.5", precision 0.01") shows the triangles shrinking onto the images.
The GIF plays the same panels one step per frame.
"""
t0 = time.perf_counter()
teaching_solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.5),
    pixel_scale_precision=0.01,
    magnification_threshold=0.1,
)
tracer = al.Tracer(galaxies=[tm.lens(), tm.source()])
for animate in (False, True):
    aplt.plot_point_solver(
        solver=teaching_solver,
        tracer=tracer,
        source_plane_coordinate=tm.SOURCE_CENTRE,
        positions=dataset.positions,
        output_path=image_path / "teaching",
        output_format="png",
        animate=animate,
    )
print(f"teaching point_solver: {time.perf_counter() - t0:.1f}s")
print(
    f"Wrote {sorted(p.name for p in image_path.iterdir())} under {image_path.relative_to(REPO_ROOT)}"
)
