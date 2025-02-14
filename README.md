# Magneton

Repo for substructure-aware representation learning and benchmarking project.

## Setup

Install using [`uv`](https://docs.astral.sh/uv/). Given `uv` installed using instructions from their website, install this package as:
```
uv sync
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