"""Pinned Lenstool catalogue and mass model, and HFF F160W sky registration."""

import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import autolens as al
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

CACHE = Path(__file__).resolve().parents[2] / "output" / "macs_j0416_inputs"
BASE = "https://www.fe.infn.it/astro/lensing/M0416_Bergamini22/"
FITS = "hlsp_frontier_hst_wfc3-60mas_macs0416_f160w_v1.0-epoch2_drz.fits"
SOURCES = {
    "best.par": (
        BASE + "best.par",
        "16df3edf1ccc5a1edc17205fbdaacd2f36b37f91102124afea2e96b75c98041f",
    ),
    "obs_arcs.cat": (
        BASE + "obs_arcs.cat",
        "bb70861cd29ab1b768755311375a633891d54cf2164207e54b7ad6d79732b60e",
    ),
    "chires.dat": (
        BASE + "chires.dat",
        "0af67a6765ba3af324243886bc32d513cd3d3718e746bcb3728f41f572bb0d53",
    ),
    FITS: (
        "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/images/hst/v1.0-epoch2/" + FITS,
        "5c974843f9e66400000c724401249de0ef42f1a988eb199fd04abf697b3327b1",
    ),
}
FIELD = (-52, 63, -77, 38)


def verified_input(name):
    url, expected = SOURCES[name]
    path = CACHE / name

    def verify(candidate):
        with candidate.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {name} from {url}")

    if path.exists():
        verify(path)
        return path
    CACHE.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(dir=CACHE, delete=False) as f:
        temporary = Path(f.name)
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        verify(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_model():
    text = verified_input("best.par").read_text()
    ra0, dec0 = map(float, re.search(r"reference\s+3\s+([\d.]+)\s+([-\d.]+)", text).groups())
    families = defaultdict(list)
    for line in verified_input("obs_arcs.cat").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ident, ra, dec, a, b, _, z, _ = line.split()
        m = re.fullmatch(r"(\d+(?:\.\d+)?)[a-z]", ident)
        if not m or float(a) != float(b):
            raise ValueError(f"Unexpected observed image: {line}")
        y = (float(dec) - dec0) * 3600
        x = -(float(ra) - ra0) * np.cos(np.deg2rad(dec0)) * 3600
        families[m[1]].append((ident, y, x, float(a), float(z)))
    zmax = max(rows[0][4] for rows in families.values())
    cosmology = dict(H0=70.0, Om0=0.3)
    lenses = []
    for block in re.findall(r"potentiel\s+\S+\s*\n(.*?)\bend\b", text, re.S):
        v = {row[0]: float(row[1]) for line in block.splitlines() if len(row := line.split()) == 2}
        if v["profil"] != 81:
            raise ValueError("Expected Lenstool dPIE profile 81")
        z = v["z_lens"]
        mass = al.mp.dPIEMass(
            centre=(v["y_centre"], v["x_centre"]),
            ellipticity=v["ellipticite"],
            angle_pos=v["angle_pos"],
            sigma=v["v_disp"],
            r_core=v["core_radius"],
            r_cut=v["cut_radius"],
            redshift_object=z,
            redshift_source=zmax,
            **cosmology,
        )
        lenses.append(al.Galaxy(redshift=z, mass=mass))
    datasets = {}
    sources = []
    for name, rows in families.items():
        ids, y, x, sigma, z = zip(*rows, strict=True)
        datasets[name] = al.PointDataset(
            name=name,
            positions=list(zip(y, x)),
            positions_noise_map=list(sigma),
            redshift=z[0],
            image_ids=list(ids),
        )
        sources.append(al.Galaxy(redshift=z[0], **{name: al.ps.PointSolved()}))
    offsets = {}
    for line in verified_input("chires.dat").read_text().splitlines():
        row = line.split()
        if len(row) == 13 and row[0].isdigit() and row[3] == "1":
            if row[1] in offsets:
                raise ValueError(f"Duplicate residual for {row[1]}")
            offsets[row[1]] = (float(row[11]), -float(row[10]))
    ids = [i for ds in datasets.values() for i in ds.image_ids]
    if (len(lenses), len(sources), len(ids)) != (222, 88, 237) or set(ids) != set(offsets):
        raise ValueError("Unexpected coverage in MACS J0416 tables")
    return (
        datasets,
        al.Tracer(galaxies=lenses + sources, cosmology=al.cosmo.FlatLambdaCDM(**cosmology)),
        offsets,
        (ra0, dec0),
    )


class Mosaic:
    def __init__(self, reference):
        self.pixels, header = fits.getdata(verified_input(FITS), header=True)
        self.wcs = WCS(header).celestial
        self.ra, self.dec = reference

    def raster(self, bounds, step):
        x0, x1, y0, y1 = bounds
        ny, nx = int(np.ceil((y1 - y0) / step)), int(np.ceil((x1 - x0) / step))
        y, x = np.meshgrid(
            y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny,
            x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx,
            indexing="ij",
        )
        px, py = self.wcs.world_to_pixel_values(
            self.ra - x / (3600 * np.cos(np.radians(self.dec))), self.dec + y / 3600
        )
        return map_coordinates(
            self.pixels, [py, px], order=1, mode="constant", cval=np.nan, prefilter=False
        )
