"""
Visualization: Weak Lensing
===========================

Renders the two figures ``VisualizerWeak`` writes for a weak-lensing fit (a shear catalogue
around a cluster-scale lens), so the latest version of each lives in git (``GALLERY.md``).

Output tree (wiped at the start of every run)::

    scripts/weak/images/visualization/
        dataset.png   <- visualize_before_fit (the plotter writes subplot_weak_dataset.png; renamed)
        fit.png       <- visualize(quick_update=True) (writes subplot_fit_weak.png; renamed)

Every other file the visualizer writes is deleted: this domain tracks only the dataset and the
fit (bespoke weak-lensing figures are a follow-up).

Dataset: ``dataset/weak/simple`` (1500 shears in a 50"-200" annulus, shape noise 0.25), from
``scripts/misc/simulators/weak.py``. The model is that simulator's TRUE model
(``true_models/weak.py``).

Run from the repo root::

    python scripts/weak/visualization.py
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
    output_path=str(REPO_ROOT / "scripts" / "weak" / "images"),
)

import autofit as af
import autolens as al
from autolens.weak.model.visualizer import VisualizerWeak

from _viz_cli import ensure_dataset, keep_dataset_and_fit
from true_models import weak as tm

STEM = Path(__file__).stem

"""
__Dataset__
"""
dataset_path = ensure_dataset("weak", workspace_root=REPO_ROOT)
dataset = al.from_json(file_path=dataset_path / "dataset.json")

"""
__True Model__

The simulator's lens and source, every parameter fixed: the prior-median instance IS the true
model.
"""
model = af.Collection(galaxies=af.Collection(lens=tm.lens(), source=tm.source()))

analysis = al.AnalysisWeak(dataset=dataset, use_jax=False)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "weak" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch = REPO_ROOT / "output" / "visualization" / "weak" / STEM
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)

paths = SimpleNamespace(image_path=image_path, output_path=scratch)

"""
__Visualize__
"""
t0 = time.perf_counter()
VisualizerWeak.visualize_before_fit(analysis=analysis, paths=paths, model=model)
VisualizerWeak.visualize(
    analysis=analysis,
    paths=paths,
    instance=model.instance_from_prior_medians(),
    during_analysis=False,
    quick_update=True,
)
keep_dataset_and_fit(
    image_path, dataset_png="subplot_weak_dataset.png", fit_png="subplot_fit_weak.png"
)
print(f"visualize: {time.perf_counter() - t0:.1f}s")
print(
    f"Wrote {sorted(p.name for p in image_path.iterdir())} under {image_path.relative_to(REPO_ROOT)}"
)
