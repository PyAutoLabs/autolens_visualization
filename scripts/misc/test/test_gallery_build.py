"""Hermetic tests for ``gallery/gallery_build.py``.

A fabricated ``scripts/alpha/images/visualization/{a.png, sub/b.png}`` tree in a tmp
dir stands in for real producer output. No PyAutoLens import (the version header is
passed explicitly and ``stack_versions`` is stubbed), no real figures: the point is
the tracked ``gallery/viz_manifest.yaml`` shape, the relative links in GALLERY.md, and
that ``--check`` catches a figure set or figure content that drifted from the
committed GALLERY.md / manifest.

The module is loaded straight off its path (``gallery/`` is not a package).

Run::

    cd autolens_visualization
    python -m pytest scripts/misc/test/test_gallery_build.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys as _sys
from pathlib import Path as _Path

import pytest
import yaml


def _repo_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_visualization root (ruff.toml) not found")


ROOT = _repo_root()
_MODULE_PATH = ROOT / "gallery" / "gallery_build.py"
_spec = importlib.util.spec_from_file_location("gallery_build_under_test", _MODULE_PATH)
gb = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = gb
_spec.loader.exec_module(gb)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"  # a header is enough: the builder never decodes


@pytest.fixture
def tree(tmp_path):
    img = tmp_path / "scripts" / "alpha" / "images" / "visualization"
    (img / "sub").mkdir(parents=True)
    (img / "a.png").write_bytes(PNG_BYTES)
    (img / "sub" / "b.png").write_bytes(PNG_BYTES)
    return tmp_path


def _build(root, *extra):
    return gb.main(["--root", str(root), "--version-string", "9.9.9", *extra])


FAKE_STACK = {"autolens": "9.9.9", "autogalaxy": "1.0", "autoarray": "1.0", "autofit": "1.0"}


@pytest.fixture(autouse=True)
def _no_stack_import(monkeypatch):
    """Keep the tests hermetic: never import the PyAuto stack for ``rendered_with``."""
    monkeypatch.setattr(
        gb, "stack_versions", lambda override=None: {**FAKE_STACK, "autolens": override or "9.9.9"}
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_tracked_manifest_content(tree):
    assert _build(tree) == 0
    doc = yaml.safe_load((tree / "gallery" / "viz_manifest.yaml").read_text())
    assert doc["schema"] == 1
    assert doc["rendered_with"] == FAKE_STACK
    assert doc["generated"]
    assert doc["figure_count"] == 2
    assert doc["figures"] == [
        {
            "file": "scripts/alpha/images/visualization/a.png",
            "producer": "scripts/alpha/visualization.py",
            "domain": "alpha",
            "source": "",
            "bytes": len(PNG_BYTES),
            "sha256": _sha(PNG_BYTES),
        },
        {
            "file": "scripts/alpha/images/visualization/sub/b.png",
            "producer": "scripts/alpha/visualization.py",
            "domain": "alpha",
            "source": "sub",
            "bytes": len(PNG_BYTES),
            "sha256": _sha(PNG_BYTES),
        },
    ]
    # The gitignored copy the Eyes conductor looks for is byte-identical.
    assert (tree / "output" / "gallery" / "viz_manifest.yaml").read_text() == (
        tree / "gallery" / "viz_manifest.yaml"
    ).read_text()
    assert (tree / "output" / "gallery" / "gallery.html").is_file()


def test_manifest_lists_png_figures_only(tree):
    (tree / "scripts" / "alpha" / "images" / "visualization" / "data.fits").write_bytes(b"x")
    assert _build(tree) == 0
    doc = yaml.safe_load((tree / "gallery" / "viz_manifest.yaml").read_text())
    assert [f["file"].rsplit(".", 1)[-1] for f in doc["figures"]] == ["png", "png"]


def test_rebuild_of_unchanged_tree_is_byte_identical(tree):
    assert _build(tree) == 0
    path = tree / "gallery" / "viz_manifest.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["generated"] = "2000-01-01"
    path.write_text(gb.dump_manifest(doc))
    before = path.read_text()
    assert _build(tree) == 0
    # Same figures + same stack: the old date is carried over, nothing to commit.
    assert path.read_text() == before


def test_new_stack_version_restamps_manifest(tree):
    assert _build(tree) == 0
    path = tree / "gallery" / "viz_manifest.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["generated"] = "2000-01-01"
    path.write_text(gb.dump_manifest(doc))
    assert gb.main(["--root", str(tree), "--version-string", "10.0.0"]) == 0
    doc = yaml.safe_load(path.read_text())
    assert doc["rendered_with"]["autolens"] == "10.0.0"
    assert doc["generated"] != "2000-01-01"


def test_check_fails_on_changed_png_bytes(tree):
    assert _build(tree) == 0
    # Same figure set (GALLERY.md still current), different content: stale manifest.
    (tree / "scripts" / "alpha" / "images" / "visualization" / "a.png").write_bytes(
        PNG_BYTES + b"changed"
    )
    assert _build(tree, "--check") == 1
    assert _build(tree) == 0
    assert _build(tree, "--check") == 0


def test_check_fails_without_tracked_manifest(tree):
    assert _build(tree) == 0
    (tree / "gallery" / "viz_manifest.yaml").unlink()
    assert _build(tree, "--check") == 1


def test_check_never_rewrites_tracked_manifest(tree):
    assert _build(tree) == 0
    path = tree / "gallery" / "viz_manifest.yaml"
    before = path.read_text()
    (tree / "scripts" / "alpha" / "images" / "visualization" / "c.png").write_bytes(PNG_BYTES)
    _build(tree, "--check")
    assert path.read_text() == before


def test_gallery_md_links_and_header(tree):
    assert _build(tree) == 0
    md = (tree / "GALLERY.md").read_text()
    assert "![a](scripts/alpha/images/visualization/a.png)" in md
    assert "![b](scripts/alpha/images/visualization/sub/b.png)" in md
    assert "Rendered with PyAutoLens `9.9.9` · 2 figures." in md
    assert "## alpha" in md
    assert "### alpha / visualization" in md
    assert "#### alpha / visualization / sub/" in md
    assert "- [alpha](#alpha)" in md


def test_check_passes_after_build_and_ignores_version(tree):
    assert _build(tree) == 0
    assert _build(tree, "--check") == 0
    # A different version stamp alone is not staleness.
    assert gb.main(["--root", str(tree), "--version-string", "0.0.1", "--check"]) == 0


def test_check_fails_after_extra_png(tree):
    assert _build(tree) == 0
    (tree / "scripts" / "alpha" / "images" / "visualization" / "c.png").write_bytes(PNG_BYTES)
    assert _build(tree, "--check") == 1
    # Rebuilding brings GALLERY.md back in line.
    assert _build(tree) == 0
    assert _build(tree, "--check") == 0


def test_check_fails_without_gallery_md(tree):
    assert _build(tree, "--check") == 1


def test_check_never_rewrites_gallery_md(tree):
    assert _build(tree) == 0
    before = (tree / "GALLERY.md").read_text()
    (tree / "scripts" / "alpha" / "images" / "visualization" / "c.png").write_bytes(PNG_BYTES)
    _build(tree, "--check")
    assert (tree / "GALLERY.md").read_text() == before


@pytest.fixture
def multi_tree(tree):
    """``alpha`` gains a variant producer and ``beta`` per-sub-dataset folders."""
    variant = tree / "scripts" / "alpha" / "images" / "visualization_variant"
    variant.mkdir(parents=True)
    (variant / "dataset.png").write_bytes(PNG_BYTES)
    (variant / "fit.png").write_bytes(PNG_BYTES + b"fit")
    for band in ("g", "r"):
        band_dir = tree / "scripts" / "beta" / "images" / "visualization" / band
        band_dir.mkdir(parents=True)
        (band_dir / "dataset.png").write_bytes(PNG_BYTES)
        (band_dir / "fit.png").write_bytes(PNG_BYTES)
    return tree


def test_multiple_producers_per_domain_in_manifest(multi_tree):
    assert _build(multi_tree) == 0
    doc = yaml.safe_load((multi_tree / "gallery" / "viz_manifest.yaml").read_text())
    assert doc["figure_count"] == 8
    by_file = {f["file"]: f for f in doc["figures"]}
    variant = by_file["scripts/alpha/images/visualization_variant/fit.png"]
    assert variant["producer"] == "scripts/alpha/visualization_variant.py"
    assert variant["domain"] == "alpha"
    assert variant["source"] == ""
    assert (
        by_file["scripts/alpha/images/visualization/a.png"]["producer"]
        == "scripts/alpha/visualization.py"
    )
    band = by_file["scripts/beta/images/visualization/r/fit.png"]
    assert band["producer"] == "scripts/beta/visualization.py"
    assert band["source"] == "r"
    # Domain, then producer, then file order.
    assert [f["producer"] for f in doc["figures"]] == [
        "scripts/alpha/visualization.py",
        "scripts/alpha/visualization.py",
        "scripts/alpha/visualization_variant.py",
        "scripts/alpha/visualization_variant.py",
    ] + ["scripts/beta/visualization.py"] * 4


def test_multiple_producers_per_domain_in_gallery_md(multi_tree):
    assert _build(multi_tree) == 0
    md = (multi_tree / "GALLERY.md").read_text()
    assert "### alpha / visualization" in md
    assert "### alpha / visualization_variant" in md
    assert "#### alpha / visualization_variant / top level" in md
    assert "#### beta / visualization / g/" in md
    assert "#### beta / visualization / r/" in md
    assert "![fit](scripts/alpha/images/visualization_variant/fit.png)" in md
    assert md.count("## alpha\n") == 1
    assert _build(multi_tree, "--check") == 0


def test_check_fails_when_a_variant_producer_drops_a_figure(multi_tree):
    assert _build(multi_tree) == 0
    (multi_tree / "scripts" / "alpha" / "images" / "visualization_variant" / "fit.png").unlink()
    assert _build(multi_tree, "--check") == 1
