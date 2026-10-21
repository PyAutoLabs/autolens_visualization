"""
Visualization: Multi-Dataset
============================

Renders the two figures ``VisualizerImaging`` writes for each band of a multi-dataset fit (the
same lens observed in the g and r bands at different resolutions), so the latest version of
each lives in git (``GALLERY.md``). A multi-dataset model-fit visualizes each dataset's
analysis separately, so this producer calls the visualizer once per band.

Output tree (wiped at the start of every run)::

    scripts/multi_dataset/images/visualization/
        g/dataset.png, g/fit.png   <- visualize_before_fit, visualize(quick_update=True)
        r/dataset.png, r/fit.png

Every other file the visualizer writes is deleted: this domain tracks only the dataset and the
fit of each band.

Dataset: ``dataset/multi_dataset/lens_sersic`` (g: 0.08"/pix, r: 0.12"/pix, both 80x80,
circular mask 3.0"), from ``scripts/misc/simulators/multi_dataset.py``. The model is that
simulator's TRUE model for each band (``true_models/multi_dataset.py``).

Run from the repo root::

    python scripts/multi_dataset/visualization.py
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
    output_path=str(REPO_ROOT / "scripts" / "multi_dataset" / "images"),
)

import autofit as af
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging

from _viz_cli import ensure_dataset, keep_dataset_and_fit
from true_models import multi_dataset as tm

STEM = Path(__file__).stem

dataset_path = ensure_dataset("multi_dataset", workspace_root=REPO_ROOT)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "multi_dataset" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch_root = REPO_ROOT / "output" / "visualization" / "multi_dataset" / STEM
if scratch_root.exists():
    shutil.rmtree(scratch_root)

"""
__Per Band__

Each band is loaded, masked (circular 3.0") and over-sampled as
``autolens_workspace/scripts/multi_dataset/modeling.py`` does, and fitted with its own true
model (shared mass and shear, per-band light).
"""
for band in tm.BANDS:
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / f"{band}_data.fits",
        psf_path=dataset_path / f"{band}_psf.fits",
        noise_map_path=dataset_path / f"{band}_noise_map.fits",
        pixel_scales=tm.PIXEL_SCALES[band],
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=tm.MASK_RADIUS,
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    model = af.Collection(
        galaxies=af.Collection(
            lens=tm.lens(band), extra_galaxy=tm.extra_galaxy(band), source=tm.source(band)
        ),
        fields=af.Collection(field=tm.field()),
    )
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    band_path = image_path / band
    band_path.mkdir(parents=True)
    scratch = scratch_root / band
    scratch.mkdir(parents=True)
    paths = SimpleNamespace(image_path=band_path, output_path=scratch)

    t0 = time.perf_counter()
    VisualizerImaging.visualize_before_fit(analysis=analysis, paths=paths, model=model)
    VisualizerImaging.visualize(
        analysis=analysis,
        paths=paths,
        instance=model.instance_from_prior_medians(),
        during_analysis=False,
        quick_update=True,
    )
    keep_dataset_and_fit(band_path)
    print(f"visualize [{band}]: {time.perf_counter() - t0:.1f}s")

n_png = len(list(image_path.rglob("*.png")))
print(f"Wrote {n_png} PNGs under {image_path.relative_to(REPO_ROOT)}")
