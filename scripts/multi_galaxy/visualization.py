"""
Visualization: Multi-Galaxy
===========================

Renders the two figures ``VisualizerImaging`` writes for a multi-galaxy-scale lens (two lens
galaxies, an external shear, a light-only extra galaxy and a source), so the latest version of
each lives in git (``GALLERY.md``).

Output tree (wiped at the start of every run)::

    scripts/multi_galaxy/images/visualization/
        dataset.png   <- visualize_before_fit
        fit.png       <- visualize(quick_update=True)

Every other file the visualizer writes is deleted: this domain tracks only the dataset and the
fit.

Dataset: ``dataset/multi_galaxy/simple`` (0.05"/pix, 140x140, circular mask 3.0"), from
``scripts/misc/simulators/multi_galaxy.py``. The model is that simulator's TRUE model
(``true_models/multi_galaxy.py``).

Run from the repo root::

    python scripts/multi_galaxy/visualization.py
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
    output_path=str(REPO_ROOT / "scripts" / "multi_galaxy" / "images"),
)

import autofit as af
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging

from _viz_cli import ensure_dataset, keep_dataset_and_fit
from true_models import multi_galaxy as tm

STEM = Path(__file__).stem

"""
__Dataset__

Loaded, masked and over-sampled as ``autolens_workspace/scripts/multi_galaxy/modeling.py``
does (circular 3.0" mask, radial-bin over-sampling centred on both lens galaxies).
"""
dataset_path = ensure_dataset("multi_galaxy", workspace_root=REPO_ROOT)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=tm.PIXEL_SCALE,
)
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=tm.MASK_RADIUS
)
dataset = dataset.apply_mask(mask=mask)
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=tm.MAIN_LENS_CENTRES,
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__True Model__

The simulator's galaxies, every parameter fixed: the prior-median instance IS the true model.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens_0=tm.lens_0(), lens_1=tm.lens_1(), extra_galaxy=tm.extra_galaxy(), source=tm.source()
    ),
    fields=af.Collection(field=tm.field()),
)

analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "multi_galaxy" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch = REPO_ROOT / "output" / "visualization" / "multi_galaxy" / STEM
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)

paths = SimpleNamespace(image_path=image_path, output_path=scratch)

"""
__Visualize__
"""
t0 = time.perf_counter()
VisualizerImaging.visualize_before_fit(analysis=analysis, paths=paths, model=model)
VisualizerImaging.visualize(
    analysis=analysis,
    paths=paths,
    instance=model.instance_from_prior_medians(),
    during_analysis=False,
    quick_update=True,
)
keep_dataset_and_fit(image_path)
print(f"visualize: {time.perf_counter() - t0:.1f}s")
print(
    f"Wrote {sorted(p.name for p in image_path.iterdir())} under {image_path.relative_to(REPO_ROOT)}"
)
