"""Shared helpers for the visualization producers and simulators.

A deliberately small subset of ``autolens_profiling/_profile_cli.py``: this repo
renders figures, it does not time anything, so only the pieces the simulators and
the ``scripts/<domain>/visualization.py`` producers need are kept:

- ``repo_root()`` — walk up from any file to the directory holding ``ruff.toml``
  (the depth-proof repo-root sentinel), so scripts work from any nesting level.
- ``bootstrap_sys_path()`` — put the repo root and ``scripts/misc/`` on
  ``sys.path`` so ``instruments`` / ``_viz_cli`` / ``simulators`` import by name.
- ``dataset_path(dataset_type, instrument)`` — ``<root>/dataset/<type>/<instrument>``.
- ``auto_simulate_if_missing(...)`` — shell out to
  ``scripts/misc/simulators/<type>.py --instrument <name>`` when a dataset is absent.
- ``DATASETS`` / ``ensure_dataset(key)`` — the registry of every tracked dataset (its
  directory, simulator and the sentinel file proving it exists) and the one-call
  "path, simulating it first if absent" helper the newer producers use.
- ``mirror_to_scratch(domains)`` — copy ``scripts/<domain>/images/`` PNGs to the opt-in
  ``$PYAUTO_EYES_SCRATCH/<lib>/<domain>/`` for on-the-fly review (no-op when unset);
  also ``python _viz_cli.py scratch [domain ...]``. ``gallery/gallery_run.sh`` calls it.

Typical use at the top of a producer::

    import sys
    from pathlib import Path

    for _p in Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            sys.path.insert(0, str(_p))
            break
    from _viz_cli import auto_simulate_if_missing, dataset_path, repo_root
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the autolens_visualization root (the directory containing ``ruff.toml``)."""
    here = Path(start if start is not None else __file__).resolve()
    for p in [here, *here.parents]:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_visualization root (ruff.toml) not found")


def bootstrap_sys_path() -> Path:
    """Put the repo root and ``scripts/misc`` on ``sys.path``; return the root."""
    root = repo_root()
    for p in (root, root / "scripts" / "misc"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root


def dataset_path(dataset_type: str, instrument: str) -> Path:
    """``<root>/dataset/<dataset_type>/<instrument>``."""
    return repo_root() / "dataset" / dataset_type / instrument


# Every tracked dataset: key -> (dataset/<type>/<name>, simulator stem, sentinel file).
# Keys are the dataset-type domains, except ``point_source``: one dataset shared by the
# ``point_source_image_chi_squared`` and ``point_source_source_chi_squared`` producers.
# The simulator is ``scripts/misc/simulators/<stem>.py --instrument <name>``; for the
# dataset types without instrument presets the "instrument" is the dataset name. The
# sentinel is the file whose presence means "the dataset is there, do not simulate".
DATASETS: dict[str, dict[str, str]] = {
    "imaging": {"type": "imaging", "name": "hst", "simulator": "imaging", "sentinel": "data.fits"},
    "interferometer": {
        "type": "interferometer",
        "name": "sma",
        "simulator": "interferometer",
        "sentinel": "data.fits",
    },
    "interferometer_datacube": {
        "type": "interferometer",
        "name": "datacube",
        "simulator": "datacube",
        "sentinel": "channel_002/data.fits",
    },
    "point_source": {
        "type": "point_source",
        "name": "simple",
        "simulator": "point_source",
        "sentinel": "data.fits",
    },
    "multi_galaxy": {
        "type": "multi_galaxy",
        "name": "simple",
        "simulator": "multi_galaxy",
        "sentinel": "data.fits",
    },
    "group": {"type": "group", "name": "simple", "simulator": "group", "sentinel": "data.fits"},
    "cluster": {
        "type": "cluster",
        "name": "simple",
        "simulator": "cluster",
        "sentinel": "point_dataset_1.json",
    },
    "multi_dataset": {
        "type": "multi_dataset",
        "name": "lens_sersic",
        "simulator": "multi_dataset",
        "sentinel": "r_data.fits",
    },
    "weak": {"type": "weak", "name": "simple", "simulator": "weak", "sentinel": "dataset.json"},
}


def ensure_dataset(key: str, workspace_root: Path | None = None) -> Path:
    """Return the ``DATASETS[key]`` directory, running its simulator first if it is absent."""
    spec = DATASETS[key]
    root = workspace_root if workspace_root is not None else repo_root()
    path = root / "dataset" / spec["type"] / spec["name"]
    auto_simulate_if_missing(
        path,
        dataset_type=spec["simulator"],
        instrument=spec["name"],
        workspace_root=root,
        sentinel=spec["sentinel"],
    )
    return path


def auto_simulate_if_missing(
    dataset_dir: Path,
    *,
    dataset_type: str,
    instrument: str,
    workspace_root: Path | None = None,
    sentinel: str = "data.fits",
) -> None:
    """If ``dataset_dir`` is missing, run the matching simulator script.

    ``dataset_type`` maps to ``scripts/misc/simulators/<dataset_type>.py``
    (``imaging`` / ``interferometer`` / any ``DATASETS`` simulator stem).

    The gate is a plain ``sentinel`` (default ``data.fits``) existence check, deliberately NOT
    ``al.util.dataset.should_simulate``: that helper deletes datasets it judges to
    belong to the other ``PYAUTO_SMALL_DATASETS`` regime, and this repo's datasets
    are tracked in git so every figure is reproducible — a harness run with the
    env var set must never wipe them.
    """
    if (Path(dataset_dir) / sentinel).exists():
        return

    root = workspace_root if workspace_root is not None else repo_root()
    simulator_script = root / "scripts" / "misc" / "simulators" / f"{dataset_type}.py"
    if not simulator_script.exists():
        raise FileNotFoundError(
            f"Auto-simulate could not find simulator script at {simulator_script}."
        )

    print(
        f"  [auto-simulate] {dataset_dir} missing; invoking "
        f"scripts/misc/simulators/{dataset_type}.py --instrument {instrument}"
    )
    subprocess.run(
        [
            sys.executable,
            str(simulator_script),
            "--instrument",
            instrument,
            "--output-root",
            str(root),
        ],
        check=True,
    )


def keep_dataset_and_fit(
    image_dir: Path,
    *,
    dataset_png: str = "dataset.png",
    fit_png: str = "fit.png",
    extra_pngs: tuple[str, ...] = (),
) -> list[Path]:
    """Reduce a visualizer's output folder to exactly ``dataset.png`` + ``fit.png``.

    ``dataset_png`` / ``fit_png`` name the files the plotter actually wrote (e.g. the point
    plotter's ``dataset_point.png``); they are renamed to ``dataset.png`` / ``fit.png``. Every
    other file (other PNGs, FITS, CSV, JSON) and any sub-folder is deleted, so the newer
    domains track only the two figures. ``extra_pngs`` names further figures to keep under their
    own names (e.g. the point plotter's ``image_with_points.png`` and
    ``fit_image_with_points.png``). Raises if any expected figure is missing.
    """
    image_dir = Path(image_dir)
    kept = []
    for written, final in ((dataset_png, "dataset.png"), (fit_png, "fit.png")):
        src = image_dir / written
        if not src.is_file():
            raise FileNotFoundError(f"expected visualizer output {src} was not written")
        dst = image_dir / final
        if src != dst:
            src.replace(dst)
        kept.append(dst)
    for written in extra_pngs:
        src = image_dir / written
        if not src.is_file():
            raise FileNotFoundError(f"expected visualizer output {src} was not written")
        kept.append(src)
    for p in sorted(image_dir.iterdir()):
        if p in kept:
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    return kept


# The PyAutoEyes dashboard names this repo's section by its library import name, and the
# scratch mirror reuses that name: ``$PYAUTO_EYES_SCRATCH/<EYES_LIB>/<domain>/``.
EYES_LIB = "autolens"
EYES_SCRATCH_ENV = "PYAUTO_EYES_SCRATCH"


def mirror_to_scratch(domains: list[str] | None = None) -> Path | None:
    """Copy freshly rendered PNGs (and GIF animations) to the opt-in Eyes scratch folder.

    Returns the scratch lib folder.

    For on-the-fly figure review before any PR or CI. When ``$PYAUTO_EYES_SCRATCH`` is
    unset this is a silent no-op returning ``None`` (so CI and other machines are
    untouched). Otherwise each domain's ``$PYAUTO_EYES_SCRATCH/<EYES_LIB>/<domain>/`` is
    deleted and refilled with every PNG and GIF under ``scripts/<domain>/images/`` (subpaths
    below ``images/`` kept), so no stale figure survives a re-render. ``domains=None``
    mirrors every domain that has an ``images/`` folder (``misc`` never does).
    """
    scratch = os.environ.get(EYES_SCRATCH_ENV)
    if not scratch:
        return None
    scripts = repo_root() / "scripts"
    if domains is None:
        domains = sorted(p.parent.name for p in scripts.glob("*/images") if p.parent.name != "misc")
    lib_dir = Path(scratch).expanduser() / EYES_LIB
    for domain in domains:
        images = scripts / domain / "images"
        if not images.is_dir():
            print(f"  [eyes-scratch] skip {domain}: no {images.relative_to(scripts.parent)}")
            continue
        dest = lib_dir / domain
        shutil.rmtree(dest, ignore_errors=True)
        figures = sorted([*images.rglob("*.png"), *images.rglob("*.gif")])
        for figure in figures:
            target = dest / figure.relative_to(images)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure, target)
        print(f"  [eyes-scratch] {len(figures)} figures (PNG/GIF) -> {dest}")
    return lib_dir


if __name__ == "__main__":
    # ``python _viz_cli.py scratch [domain ...]`` — mirror rendered PNGs to the Eyes scratch.
    if len(sys.argv) < 2 or sys.argv[1] != "scratch":
        sys.exit("usage: python _viz_cli.py scratch [domain ...]")
    if mirror_to_scratch(sys.argv[2:] or None) is None:
        print(f"  [eyes-scratch] ${EYES_SCRATCH_ENV} is unset; nothing mirrored")
