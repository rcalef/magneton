# Magneton

Repo for substructure-aware representation learning and benchmarking project.

## Setup

Install using [`uv`](https://docs.astral.sh/uv/). Given `uv` installed using instructions from their website, install this package as:
```
# Note that installing torch-cluster (required by torchdrug) requires GCC 9 or later.
# On OpenMind, can run the following:
#  module load openmind/mpc/1.2.1 openmind/mpfr/4.1.0 openmind/isl/0.23 openmind/gcc/10.3.0
# Also note that install can take a long time (~10 mins) because of torch-drug
uv sync --extra build
uv sync --extra build --extra compile
uv pip install -e .
source .venv/bin/activate
```

## Experiments

# Run full pipeline
```
python -m magneton.cli

```

# Run specific stages
```
python -m magneton.cli stage=embed
python -m magneton.cli stage=train
python -m magneton.cli stage=visualize
```
