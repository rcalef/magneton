# Magneton

Repo for substructure-aware representation learning and benchmarking project.

## Setup

Install using [`uv`](https://docs.astral.sh/uv/). Given `uv` installed using instructions from their website, install this package as:
```
# Note that installing torch-cluster (required by torchdrug) requires GCC 9 or later.
# On OpenMind, can run the following:
# module load openmind/mpc/1.2.1 
# module load openmind/mpfr/4.1.0 
# module load openmind/isl/0.23
# module load openmind/gcc/10.3.0
# Also note that install can take a long time (~10 mins) because of torch-drug
uv sync --extra build
uv sync --extra build --extra compile
uv pip install -e .
source .venv/bin/activate
```

## Experiments

# Run specific stages
```
python magneton/cli.py +stages=["embed"]
python magneton/cli.py +stages=["train"]
python magneton/cli.py +stages=["eval"]
```
