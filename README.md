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
# TODO: either remove torchdrug or make this less shitty
uv pip install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
uv pip install torchdrug
uv pip install -e .
source .venv/bin/activate
```

# Experiments

## Run specific stages
```
python magneton/cli.py +stages=["embed"]
python magneton/cli.py +stages=["train"]
python magneton/cli.py +stages=["eval"]
```

## For debugging and testing before running on wandb
```
python magneton/cli.py training.dev_run=True
```

## Datasets
See config.yaml for exact details but there is the main dataset and splits and a debug dataset.

## Run ESM-C with flash attention (if installed)
```
python magneton/cli.py embedding=esmc ++embedding.model_params.use_flash_attn=True training.batch_size=32
```

## Run with elastic weight consolidation for embedder fine-tuning
```
python magneton/cli.py \
  embedding=esmc_300m  \
  model.frozen_embedder=False \
  training.loss_strategy=ewc \
  training.ewc_weight=400 \
  training.embedding_learning_rate=1e-5 \
  training.embedding_weight_decay=0 \
  training.batch_size=24 \
  ++embedding.model_params.use_flash_attn=True
```
