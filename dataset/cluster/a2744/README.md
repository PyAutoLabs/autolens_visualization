# Pandora (Abell 2744)

The three CSVs are a snapshot of the [workspace catalogue](https://github.com/PyAutoLabs/autolens_workspace/tree/465c086eeb83b413f4ccb35a0a61a8fa34b87041/dataset/cluster/a2744). The 24 measured positions belong to seven spectroscopic families. Two individually modelled galaxies, a single NFW halo and 188 luminosity-scaled members provide the starting lens model. Each source centre is solved analytically. The mass parameters are fixed. The four overviews and seven family diagnostics use the same 0.5″ positional uncertainty as the workspace.

The cached 600 × 600 H-band [CDS hips2fits cutout](data.fits) keeps the gallery reproducible offline. Its SHA-256 is `f8f7161b84c04f2488c386bba728849648afb22bc63ef26f1a13bfbd6f0da72c`. The accompanying workspace loader uses its FITS reference pixel and axis directions to align the image with the east-positive, north-positive catalogue. When refreshing the dataset, copy the loader and its inputs together.

Credit the multiple-image and member catalogues to Bergamini et al. (2023, A&A 670, A60) and the Hubble Frontier Fields imaging to Lotz et al. (2017).
