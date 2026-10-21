"""
Visualization: Interferometer Datacube
======================================

Renders the two figures ``VisualizerInterferometer`` writes for each channel of an
interferometer datacube (one lens, a source whose brightness and centre vary with channel), so
the latest version of each lives in git (``GALLERY.md``). A datacube model-fit visualizes each
channel's analysis separately, so this producer calls the visualizer once per channel.

Output tree (wiped at the start of every run)::

    scripts/interferometer/images/visualization_datacube/
        channel_0/dataset.png, channel_0/fit.png   <- visualize_before_fit, visualize(quick_update=True)
        channel_1/..., channel_2/...

Every other file the visualizer writes is deleted: this variant tracks only the dataset and
the fit of each channel. (The full figure set of a single interferometer dataset is
``visualization.py`` beside this file.)

Dataset: ``dataset/interferometer/datacube`` (three channels, the ``sma`` uv-coverage, 256x256
real-space grid at 0.1"/pix, circular mask 3.5"), from
``scripts/misc/simulators/datacube.py``. The model is that simulator's TRUE model for each
channel (``true_models/datacube.py``).

Run from the repo root::

    python scripts/interferometer/visualization_datacube.py
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
    output_path=str(REPO_ROOT / "scripts" / "interferometer" / "images"),
)

import autofit as af
import autolens as al
from autolens.interferometer.model.visualizer import VisualizerInterferometer

from _viz_cli import ensure_dataset, keep_dataset_and_fit
from instruments.interferometer import INSTRUMENTS
from true_models import datacube as tm

STEM = Path(__file__).stem
config = INSTRUMENTS["sma"]

dataset_path = ensure_dataset("datacube", workspace_root=REPO_ROOT)

real_space_mask = al.Mask2D.circular(
    shape_native=config["real_space_shape"],
    pixel_scales=config["pixel_scale"],
    radius=config["mask_radius"],
)

"""
__Paths__
"""
image_path = REPO_ROOT / "scripts" / "interferometer" / "images" / STEM
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)

scratch_root = REPO_ROOT / "output" / "visualization" / "interferometer" / STEM
if scratch_root.exists():
    shutil.rmtree(scratch_root)

"""
__Per Channel__
"""
for channel in range(tm.N_CHANNELS):
    channel_dir = dataset_path / f"channel_{channel:03d}"
    dataset = al.Interferometer.from_fits(
        data_path=channel_dir / "data.fits",
        noise_map_path=channel_dir / "noise_map.fits",
        uv_wavelengths_path=channel_dir / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )

    model = af.Collection(
        galaxies=af.Collection(lens=tm.lens(), source=tm.source(channel)),
        fields=af.Collection(field=tm.field()),
    )
    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=False)

    channel_path = image_path / f"channel_{channel}"
    channel_path.mkdir(parents=True)
    scratch = scratch_root / f"channel_{channel}"
    scratch.mkdir(parents=True)
    paths = SimpleNamespace(image_path=channel_path, output_path=scratch)

    t0 = time.perf_counter()
    VisualizerInterferometer.visualize_before_fit(analysis=analysis, paths=paths, model=model)
    VisualizerInterferometer.visualize(
        analysis=analysis,
        paths=paths,
        instance=model.instance_from_prior_medians(),
        during_analysis=False,
        quick_update=True,
    )
    keep_dataset_and_fit(channel_path)
    print(f"visualize [channel {channel}]: {time.perf_counter() - t0:.1f}s")

n_png = len(list(image_path.rglob("*.png")))
print(f"Wrote {n_png} PNGs under {image_path.relative_to(REPO_ROOT)}")
