# autolens_visualization

**[Browse the gallery → GALLERY.md](GALLERY.md)**

The permanent, rendered gallery of every figure PyAutoLens writes during a model-fit — the
**lens visualization project repo** of the PyAutoLabs organism.

This repo owns the lens figures: the producer scripts, the simulators and datasets, the
all-on [`plots.yaml`](config/visualize/plots.yaml), the instrument presets, the tracked PNGs,
[`GALLERY.md`](GALLERY.md) and the render harness. The organ
[PyAutoEyes](https://github.com/PyAutoLabs/PyAutoEyes) is the cross-project visualization
dashboard: it reads this repo's tracked figure manifest
([`gallery/viz_manifest.yaml`](gallery/viz_manifest.yaml)) and links to the PNGs here — it never
renders or copies them. Other libraries get sibling `<lib>_visualization` project repos in the
same shape.

## Vision

Until now, the only way to see what a PyAutoLens figure looks like was to run a modelling script
and open its `output/` folder. This repo keeps the most up-to-date rendering of **every**
visualizer output in git — on realistic HST-scale imaging and SMA-scale interferometer data — so
there is one place to see every figure, judge it, and improve it (by hand or in an AI chat
pointed at the plotting source) without re-running anything. Imaging and interferometer today;
multi-galaxy, group and cluster scale next.

Each figure is exactly what `VisualizerImaging` / `VisualizerInterferometer` write to a fit's
`image/` folder, rendered with the simulator's true model for a parametric (SersicCore) source and
a Delaunay pixelized source, with every toggle in
[`config/visualize/plots.yaml`](config/visualize/plots.yaml) switched on.

## Render locally

```bash
source activate.sh                  # library checkouts on PYTHONPATH (see the file)
bash gallery/gallery_run.sh --all   # run both producers, rebuild GALLERY.md + manifest, --check
```

Or one domain: `python scripts/imaging/visualization.py`, then `python gallery/gallery_build.py`.
Commit the regenerated PNGs under `scripts/<domain>/images/` together with `GALLERY.md` and
`gallery/viz_manifest.yaml` (every figure's producer, domain, source type, path, byte size and
sha256, plus the stack versions it was rendered with). On every PyAutoLens release,
[`render.yml`](.github/workflows/render.yml) re-renders with the released stack, commits the
result and pings PyAutoEyes (`repository_dispatch: eyes-refresh`) to refresh its dashboard.

## Add a domain

- Add a simulator (or instrument preset) under `scripts/misc/simulators/` and track its dataset
  under `dataset/<domain>/<instrument>/`.
- Add a flat producer `scripts/<domain>/visualization.py` modelled on
  [`scripts/imaging/visualization.py`](scripts/imaging/visualization.py).
- Run `bash gallery/gallery_run.sh --all` and commit the PNGs + `GALLERY.md` +
  `gallery/viz_manifest.yaml`.

## Improve a figure

Three edit surfaces, from cheapest to deepest:

- **Config** — [`config/visualize/plots.yaml`](config/visualize/plots.yaml): which figures are
  written at all.
- **Plot API** — the plotting code in the libraries: `PyAutoLens/autolens/**/plot/`,
  `PyAutoGalaxy/autogalaxy/**/plot/`, `PyAutoArray/autoarray/plot/`. Change it there (normal
  library workflow), then re-render here.
- **Script** — the producer in `scripts/<domain>/visualization.py` (dataset, model, source types).

The Brain Eyes agent runs the review loop on this repo:
`bin/pyauto-brain eyes survey lens/autolens_visualization`.

## Related repos

- [PyAutoEyes](https://github.com/PyAutoLabs/PyAutoEyes) — the organ: the cross-project
  visualization dashboard that aggregates this repo (reads `gallery/viz_manifest.yaml`, links to
  the PNGs here).
- [autolens_workspace](https://github.com/PyAutoLabs/autolens_workspace) — user-facing science
  scripts and tutorials.
- [autolens_workspace_test](https://github.com/PyAutoLabs/autolens_workspace_test) — visualization
  tests (file / FITS-HDU assertions) and the gallery harness this one was adapted from.
- [autolens_profiling](https://github.com/PyAutoLabs/autolens_profiling) — likelihood timing; the
  source of this repo's instrument presets, simulators and HST dataset.

## Community & Contributing

**PyAutoLens** is built in the open by its users: everyone is welcome to ask questions,
share what they have made with it, and contribute.

Questions, ideas and bug reports: the [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions).
Chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

Community-built tools and tutorials, and how to contribute: the [**PyAutoLens** community page](https://pyautolens.readthedocs.io/en/latest/general/community.html).
