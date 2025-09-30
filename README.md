# Magneton

Repo for substructure-aware representation learning and benchmarking project.

## Setup

Install using [`uv`](https://docs.astral.sh/uv/). Given `uv` installed using instructions from their website, install this package as:
```
# After cloning
git submodule update --init

# --managed-python not strictly required, but can be helpful
# on shared systems with limited user configurability (e.g.
# university SLURM clusters)
uv sync --managed-python
```

The below commands are specifically for installing Flash Attention,
which is optional, but will speed up ESM-C, ESM2,
and SaProt models. Building and configuring Flash Attention
can require different steps (e.g. loading modules) depending
on the system, so we leave it as optional.
```
uv sync --extra flash
# Below is currently required for Flash Attention support
# for SaProt and ESM2. transformers 4.56.1 updates their ESM
# implementation to support flash attention, but EvoScale esm
# is currently marked as transformers < 4.48.2. See:
#  https://github.com/evolutionaryscale/esm/issues/265
uv pip install --upgrade-package transformers transformers 
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
## Run downstream GO term classification
```
python magneton/cli.py \
  stages=["eval"] \
  evaluate=go \
 run_id="esmc_300m_orig" \
  output_dir="/home/rcalef/storage/om_storage/projects/magneton/experiments/downstream_evals/esmc_300m_orig" \
 evaluate.model_checkpoint="/weka/scratch/weka/kellislab/rcalef/projects/magneton/experiments/no_finetune/esmc_300m_domain/model_esmc_300m_domain.pt" 
 ```
