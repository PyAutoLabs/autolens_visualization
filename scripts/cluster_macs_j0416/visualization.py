"""MACS J0416 family fits and the Bergamini et al. published image-plane residuals."""

from pathlib import Path
from tempfile import TemporaryDirectory

import autolens as al
import autolens.plot as aplt
import matplotlib.pyplot as plt
import numpy as np
from inputs import FIELD, Mosaic, load_model
from published_offsets import STYLE, published_offsets


def render(output):
    datasets, tracer, offsets, reference = load_model()
    mosaic = Mosaic(reference)
    image = al.Array2D.no_mask(
        values=mosaic.raster(FIELD, 0.1)[::-1], pixel_scales=0.1, origin=(-19.5, 5.5)
    )
    positions = np.vstack([ds.positions.array for ds in datasets.values()])
    save = dict(output_path=str(output), output_format="png")
    for name in ("210.4", "101", "14.1"):
        fit = al.FitPointDataset(
            dataset=datasets[name],
            tracer=tracer,
            solver=None,
            fit_positions_cls=al.FitPositionsSourceSolved,
        )
        aplt.subplot_fit_family(
            fit,
            image=image,
            title_prefix="MACS J0416  |  Family",
            output_filename=f"source_{name}",
            **save,
        )
        if name == "210.4":
            aplt.plot_family_overview(fit, image=image, output_filename="overview", **save)
            aplt.plot_family_contributions(fit, output_filename="ranking", **save)
            aplt.plot_family_image_detail(
                fit, image_id="210.4c", output_filename="210.4c_zoom", **save
            )
            aplt.plot_family_source_plane(fit, output_filename="all_back_traced_images", **save)
        else:
            with plt.rc_context(STYLE):
                published_offsets(fit, positions, offsets, mosaic, output / f"published_{name}.png")


def main():
    output = Path(__file__).resolve().parent / "images" / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    # A failed input download or model evaluation leaves the tracked figures intact.
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary)
        render(staging)
        figures = list(staging.glob("*.png"))
        if len(figures) != 9:
            raise RuntimeError(f"Expected nine MACS J0416 figures, rendered {len(figures)}")
        stale = set(output.glob("*.png")) - {output / figure.name for figure in figures}
        for figure in figures:
            figure.replace(output / figure.name)
        for figure in stale:
            figure.unlink()


if __name__ == "__main__":
    main()
