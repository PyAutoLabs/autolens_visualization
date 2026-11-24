"""Render Pandora's seven observed families on the registered HST field.

The source-plane diagnostics use a fixed starting lens model with a solved
centre for each source. See dataset/cluster/a2744/README.md for the inputs.
Run from the gallery root with python scripts/cluster_pandora/visualization.py.
"""

import shutil
import sys
from pathlib import Path

import autolens as al
from autolens.point.plot import multi_source_plots as plots
from autolens.point.plot.family_fit_plots import plot_family_fits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "cluster" / "a2744"
sys.path.insert(0, str(DATA))
from pandora_image import image_from_fits

datasets = al.list_from_csv(file_path=DATA / "point_datasets.csv")
image = image_from_fits(DATA / "data.fits")
mass = al.galaxy_models_from_csv(file_path=DATA / "mass.csv", family="mass")
models = al.galaxy_af_models_from_csv_tables(mass)
galaxies = [model.instance_from_prior_medians() for model in models.values()]

# The scaling relation uses the walkthrough prior midpoint for sigma_ref.
members = al.galaxy_table_from_csv(file_path=DATA / "scaling_galaxies.csv")
source_redshift = max(dataset.redshift for dataset in datasets)
for centre, luminosity in zip(members.centres, members.luminosities, strict=True):
    profile = al.mp.dPIEMassSph(
        centre=tuple(centre),
        sigma=150 * luminosity**0.25,
        r_cut=5 * luminosity**0.7,
        r_core=0.0,
        redshift_object=0.308,
        redshift_source=source_redshift,
        H0=67.66,
        Om0=0.30966,
    )
    galaxies.append(al.Galaxy(redshift=0.308, mass=profile))

for dataset in datasets:
    galaxies.append(al.Galaxy(redshift=dataset.redshift, **{dataset.name: al.ps.PointSolved()}))
tracer = al.Tracer(galaxies=galaxies)
fits = [
    al.FitPointDataset(
        dataset=dataset, tracer=tracer, solver=None, fit_positions_cls=al.FitPositionsSourceSolved
    )
    for dataset in datasets
]

output = Path(__file__).resolve().parent / "images" / Path(__file__).stem
if output.exists():
    shutil.rmtree(output)
output.mkdir(parents=True)
save = dict(output_path=str(output), output_format="png")
plots.plot_dataset_positions(
    datasets, image_extent=image.geometry.extent, title_prefix="Pandora", filename="dataset", **save
)
plots.plot_image_with_points(image, datasets=datasets, title_prefix="Pandora", **save)

fit_title = "Pandora · starting model"
plots.plot_fit_positions(
    fits, curves={}, image_extent=image.geometry.extent, title_prefix=fit_title, **save
)
plots.plot_image_with_points(image, fits=fits, curves={}, title_prefix=fit_title, **save)
plot_family_fits(fits, image=image, title_prefix=fit_title, **save)
