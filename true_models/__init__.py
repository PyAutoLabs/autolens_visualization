"""The true models shared by each simulator and its visualization producer.

One module per dataset type (``point_source``, ``multi_galaxy``, ``group``, ``cluster``,
``multi_dataset``, ``weak``, ``datacube``). ``scripts/misc/simulators/<domain>.py`` builds the
dataset from these galaxies, and ``scripts/<domain>/visualization*.py`` fits the very same
galaxies, so every ``fit.png`` shows the fit of the model that generated the data.

Each module exposes plain functions returning ``al.Galaxy`` / ``al.MassField`` instances (so
importing it never touches config), plus the dataset geometry constants both sides need.
The imaging and interferometer domains predate this package and restate their models verbatim
in the simulator and producer instead.
"""
