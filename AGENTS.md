# autolens_visualization — Agent Instructions

This repo is the single home for **what PyAutoLens figures look like**: it stores, in git, the
most up-to-date rendering of every figure the PyAutoLens visualizers write during a model-fit,
on realistic HST-scale imaging and SMA-scale interferometer data — plus the dataset and fit
figures of every other PyAutoLens dataset type (point source, multi-galaxy, group, cluster,
multi-dataset, interferometer datacube, weak lensing) — so visualization can be judged
and improved (by humans or AI chats against the source) without re-running a workspace fit. It is
a collection of standalone producer scripts, **not** an installable package — there is no
`pyproject.toml`. These are the canonical, agent-agnostic instructions for this repo; the
`README.md` is the human-facing overview and `GALLERY.md` is the browsable gallery.

## Layering: project repo vs organ

This is a **project repo** (category `project`, like `autolens_profiling` and
`autolens_inference`): it makes, stores and tracks the lens figures — producers, simulators,
datasets, `plots.yaml`, instruments, tracked PNGs, `GALLERY.md`, the render harness. The organ
**PyAutoEyes** is the cross-project visualization dashboard over the `<lib>_visualization`
project repos: it reads this repo's tracked `gallery/viz_manifest.yaml` and links to the PNGs
here; it renders nothing and copies no figures. Judging figures is the Brain's Eyes conductor;
figure changes land here (producer / config) or in the libraries (plot API), never in the organ.

## Repository Structure

The `scripts/<domain>/` folders mirror the `autolens_workspace/scripts/` dataset folders (none
go in `misc/`). Producers are laid out **flat** (`scripts/<domain>/visualization*.py`), because
the Brain Eyes agent scans `scripts/<domain>/*.py` non-recursively for stems containing
`visualization`. Each producer `<stem>.py` writes `scripts/<domain>/images/<stem>/`:

```
scripts/
  imaging/visualization.py              VisualizerImaging on dataset/imaging/hst — full set:
                                        before-fit PNGs + parametric/ + delaunay/
  interferometer/visualization.py       VisualizerInterferometer on dataset/interferometer/sma
                                        — full set, same layout
  interferometer/visualization_datacube.py   per channel: channel_<i>/{dataset,fit}.png
  point_source/visualization_image_chi_squared.py   VisualizerPoint, FitPositionsImagePair
  point_source/visualization_source_chi_squared.py  VisualizerPoint, FitPositionsSource
  multi_galaxy/visualization.py         VisualizerImaging, two lenses + extra galaxy
  group/visualization.py                VisualizerImaging, main lens + two extra galaxies
  cluster/visualization.py              PlotterPoint, multi-plane: point_<i>/{dataset,fit}.png
  multi_dataset/visualization.py        VisualizerImaging per band: <band>/{dataset,fit}.png
  weak/visualization.py                 VisualizerWeak
  misc/simulators/                      one simulator per dataset (regenerate datasets)
  misc/test/                            hermetic pytest for the gallery builder
true_models/                            the true model of each newer dataset type, shared by
                                        its simulator and producer
gallery/gallery_build.py                GALLERY.md + gallery/viz_manifest.yaml + output/gallery/
gallery/viz_manifest.yaml               TRACKED, generated figure manifest (PyAutoEyes read contract)
gallery/gallery_run.sh                  run producers -> build -> --check
config/general.yaml                     layered over the library config (version check off)
config/visualize/plots.yaml             the library default with EVERY toggle on
instruments/                            imaging + interferometer presets (copied from profiling)
dataset/imaging/hst/                    TRACKED, byte-for-byte copy of autolens_profiling's hst
dataset/interferometer/sma/             TRACKED, simulated here (190 visibilities)
dataset/<domain>/<name>/                TRACKED, simulated here (see `DATASETS` in _viz_cli.py)
_viz_cli.py                             repo-root finder, DATASETS registry, auto-simulate hook,
                                        keep_dataset_and_fit pruning helper
GALLERY.md                              TRACKED, generated — never edit by hand
```

**Two figures per newer dataset type.** Imaging and interferometer (`visualization.py`) render
the full figure set. Every other producer tracks only `dataset.png` and `fit.png`: it calls
`visualize_before_fit`, then `visualize(..., quick_update=True)` with the true model, then
`_viz_cli.keep_dataset_and_fit` renames the plotter's files where they differ (point:
`dataset_point.png`; weak: `subplot_weak_dataset.png` / `subplot_fit_weak.png`) and deletes
everything else. Producers with several sub-datasets write one pair per sub-folder
(`multi_dataset` per band, the datacube per channel, `cluster` per point-source plane). Many
of these figures are the galaxy-scale plots and look poor on point / cluster / weak data —
bespoke figures are a follow-up through the Eyes dashboard. `cluster` calls the two
`PlotterPoint` methods directly instead of `VisualizerPoint`: the visualizer's critical-curve
pre-pass costs ~100 s on the 15-galaxy multi-plane lens and the point fit subplot never draws
the curves (the PNGs are byte-identical).

**What is tracked.** PNG figures under `scripts/<domain>/images/**`, `GALLERY.md` and
`gallery/viz_manifest.yaml`. The FITS /
CSV / JSON data products the visualizers also write are gitignored (bulky, not viewable on
GitHub), as are `output/` and `dataset/**/lensed_source.fits`.

**Import model.** Producers find the repo root by walking up to the directory containing
`ruff.toml` (a depth-proof sentinel) and put it on `sys.path`, so `_viz_cli` and `instruments`
import by their top-level names.

## Rendering

From the repo root, with the library checkouts on `PYTHONPATH` (`source activate.sh`):

```bash
bash gallery/gallery_run.sh --all        # every producer, then build + --check (~4 min)
python scripts/imaging/visualization.py  # one producer (~75 s)
python gallery/gallery_build.py          # rebuild GALLERY.md + gallery/viz_manifest.yaml + output/gallery/
python gallery/gallery_build.py --check  # fail if GALLERY.md or the manifest is stale vs the PNGs on disk
```

Each producer wipes its own `scripts/<domain>/images/<stem>/` tree first, so the committed
PNG set is exactly what the last run produced. Commit the PNGs, `GALLERY.md` and
`gallery/viz_manifest.yaml` together.

**The tracked manifest** (`gallery/viz_manifest.yaml`, schema 1) lists every committed PNG as
`{file, producer, domain, source, bytes, sha256}` (`source` = the per-source sub-folder,
`parametric` / `delaunay`, a band / channel / `point_<i>`, or `""` for top-level figures), plus `rendered_with:` (the
autolens / autogalaxy / autoarray / autofit versions) and `generated:` (the date the figures or
stack last changed — carried over on an unchanged rebuild, so re-running is a git no-op). There
are no mtimes: it is reproducible from a checkout. `--check` ignores only `generated:` and
`rendered_with:`; any added, removed or byte-changed PNG fails it. It is the read contract of the
PyAutoEyes dashboard — change its shape only together with the organ.

Each producer pushes `config/` via `conf.instance.push` (the all-true `plots.yaml`), loads its
tracked dataset, builds the simulator's **true model** (every parameter fixed), and calls
`Visualizer*.visualize_before_fit` once and `Visualizer*.visualize` once per source type (the
imaging and interferometer `visualization.py` producers; the others are described above)
(`parametric/` = SersicCore, `delaunay/` = Overlay image-mesh + `Delaunay` + `ConstantSplit`),
with a `SimpleNamespace(image_path=..., output_path=...)` paths stub. Adapt images are the
per-galaxy images of the parametric fit.

**Datasets.** `dataset/imaging/hst` is a byte-for-byte copy of
`autolens_profiling/dataset/imaging/hst` (do not re-simulate it; profiling documents that it does
not regenerate byte-identically). `dataset/interferometer/sma` was produced by

```bash
python scripts/misc/simulators/interferometer.py --instrument sma
```

with the imaging simulator's lens mass, shear and source (no lens light). This repo's `sma`
preset **deviates from autolens_profiling**: `noise_sigma` is 10.0 (profiling: 1000.0), so the
integrated SNR is a few hundred and the dirty image, residual and chi-squared panels show the ring
instead of pure noise. The auto-simulate hook
in `_viz_cli.py` only fires when the dataset's sentinel file is absent — it never deletes a
tracked dataset.

The newer datasets are simulated here from `scripts/misc/simulators/<name>.py`, each adapted
from the matching `autolens_workspace` simulator (`datacube.py` from
`interferometer/features/datacube/simulator.py`) and cut down to stay small, with seeded noise.
Their true models live in `true_models/<name>.py`, imported by both the simulator and the
producer, so the fit is always of the model that made the data. Each is registered in
`_viz_cli.DATASETS` (directory, simulator stem, sentinel file); producers call
`ensure_dataset("<key>")`, which simulates only when the sentinel is missing.

## Adding a domain

1. Name the domain after the `autolens_workspace/scripts/<folder>` it mirrors; never put a
   producer in `misc/`.
2. Put the true model in `true_models/<name>.py`, add a simulator
   `scripts/misc/simulators/<name>.py` (accepting `--instrument <name>` and `--output-root`)
   that builds the dataset from it, register it in `_viz_cli.DATASETS`, and track its dataset
   under `dataset/<domain>/<name>/` (keep it small: KBs to low MBs).
3. Add a flat producer `scripts/<domain>/visualization.py` modelled on an existing one, writing
   to `scripts/<domain>/images/visualization/`.
4. **Variants are extra producers, never sub-folders.** A variant of a domain (another
   likelihood, a datacube) is a sibling file `scripts/<domain>/visualization_<variant>.py`
   writing `scripts/<domain>/images/visualization_<variant>/` — the Eyes conductor scans only
   the top level of `scripts/<domain>/`, and PyAutoEyes groups figures by producer within a
   domain. Sub-datasets of one producer (bands, channels, planes) are sub-folders of its
   image dir.
5. Run `bash gallery/gallery_run.sh --all` and commit the PNGs + `GALLERY.md` +
   `gallery/viz_manifest.yaml`.

## Improving a figure (edit surfaces)

- **config** — `config/visualize/plots.yaml` here: which figures are written at all.
- **plot API** — the plotting code in the libraries: `PyAutoLens/autolens/**/plot/`,
  `PyAutoGalaxy/autogalaxy/**/plot/`, `PyAutoArray/autoarray/plot/` (library changes go through
  the normal library workflow, then this repo is re-rendered).
- **script** — the producer in `scripts/<domain>/visualization*.py` (dataset, model, source
  types) and its `true_models/` / simulator.

## Eyes contracts

Two readers depend on this repo's layout:

- **The Brain Eyes conductor** (`organs/PyAutoBrain/agents/conductors/eyes/`) reviews it:
  `bin/pyauto-brain eyes survey lens/autolens_visualization`. It expects flat
  `scripts/<domain>/visualization*.py` producers writing `scripts/<domain>/images/<stem>/**`, a
  `gallery/gallery_run.sh` harness, and `output/gallery/{gallery.html,viz_manifest.yaml}` (the
  latter a gitignored copy of the tracked manifest). Accepted critiques route through intake /
  start_dev like any other change.
- **The PyAutoEyes dashboard** reads the tracked `gallery/viz_manifest.yaml` and links to the
  PNGs; `render.yml` fires `repository_dispatch: eyes-refresh` at PyAutoLabs/PyAutoEyes after
  each release re-render.

Keep that layout when adding domains.

## Testing

The PR gate is `lint.yml` on Python 3.12 against the library mains:

```bash
ruff check .
ruff format --check .
python gallery/gallery_build.py --check
pytest scripts/misc/test -q
```

plus `lychee` on every `*.md`, then every producer runs for real followed by
`gallery_build.py --check` (a PR that changes the figure set without regenerating `GALLERY.md`
fails). `render.yml` (manual + `repository_dispatch: pyautolens-release`) re-renders with the
released PyPI stack, commits PNGs + `GALLERY.md` + `gallery/viz_manifest.yaml` back as
`github-actions[bot]` `[skip ci]`, then dispatches `eyes-refresh` to PyAutoEyes (token:
`secrets.PAT_PYAUTOLABS`; the step warns and skips when the secret is unavailable).

## Sandboxed / restricted runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/visualization.py
```

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file.**

## Related Repos

- `../PyAutoLens` — the visualizers being rendered (plus `../PyAutoGalaxy`, `../PyAutoArray`,
  `../PyAutoFit`, `../PyAutoNerves` on `PYTHONPATH`).
- `../autolens_workspace` — user-facing science scripts and tutorials.
- `../autolens_workspace_test` — visualization *tests* (file/HDU assertions) and the gallery
  harness this repo's was adapted from.
- `../autolens_profiling` — source of the instrument presets, simulators and HST dataset.
- `../../organs/PyAutoEyes` — the organ: cross-project visualization dashboard that aggregates
  this repo via `gallery/viz_manifest.yaml` (links to the PNGs, never copies them).

## Task Workflows

When changing a producer, the config or a dataset, re-render (`gallery/gallery_run.sh --all`),
keep `ruff check .` / `ruff format --check .` clean, and commit the PNGs + `GALLERY.md` +
`gallery/viz_manifest.yaml` in the same PR. Do not commit machine-specific absolute paths.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->

<!-- repos_sync:filing:begin -->
## Where to file

Questions, help with code or an analysis, ideas, bug reports and results from a
user or collaborator — or an agent acting for one — go to
<https://github.com/orgs/PyAutoLabs/discussions> in the matching category
(Help & Questions, Ideas & Proposals, Bugs & Errors, Show and tell;
Announcements is maintainers-only), never to this repo's Issues. An agent never
runs `gh issue create` for such a report: it drafts the title, category and
body and hands them to the human (sessions cannot create Discussions). Only the
development flow — Mind prompt → `/start_dev` → `/create_issue` → one issue per
task → PR — opens issues here. Why: `PyAutoMind/policy/community_surface.md`.
<!-- repos_sync:filing:end -->

<!-- repos_sync:standards:begin -->
## Shared standards

Before changing a shared interface, consult the applicable
[organism standard](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/docs/standards.md)
on demand, identify affected consumers, and validate their adoption. Change
generated guidance at its canonical source and regenerate.
<!-- repos_sync:standards:end -->
