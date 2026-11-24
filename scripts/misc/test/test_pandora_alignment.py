"""A reversed sky axis or a lost half-pixel offset displaces observed images."""

import sys
from pathlib import Path

import autolens as al
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

DATA = Path(__file__).resolve().parents[3] / "dataset" / "cluster" / "a2744"
sys.path.insert(0, str(DATA))
from pandora_image import image_from_fits


def test_hst_pixels_and_catalogue_share_the_same_sky_frame():
    path = DATA / "data.fits"
    pixels, header = fits.getdata(path, header=True)
    image = image_from_fits(path)
    np.testing.assert_array_equal(image.native, pixels[::-1, ::-1])

    families = al.list_from_csv(file_path=DATA / "point_datasets.csv")
    assert [len(family.positions) for family in families] == [3, 4, 4, 3, 3, 3, 4]
    north, east = np.vstack([family.positions for family in families]).T
    ra0, dec0 = 3.5875, -30.3972
    col, row = WCS(header).celestial.world_to_pixel_values(
        ra0 + east / (3600 * np.cos(np.radians(dec0))), dec0 + north / 3600
    )
    ny, nx = pixels.shape
    xmin, xmax, ymin, ymax = image.geometry.extent
    x = xmin + (nx - 0.5 - col) * (xmax - xmin) / nx
    y = ymax - (ny - 0.5 - row) * (ymax - ymin) / ny
    # The published small-angle catalogue approximation differs slightly from TAN.
    assert np.hypot(x - east, y - north).max() < 0.004
