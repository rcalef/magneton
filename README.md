# Magneton
<!-- [Figure 1] -->

This repository provides the code for Magneton, an integrated environment for developing substructure-aware protein models, detailed in the paper Greater than the Sum of Its Parts: Building Substructure into Protein Encoding Models. Magneton provides (1) a large-scale dataset of 530,601 proteins annotated with over 1.7 million substructures spanning 13,075 types, (2) a training framework for incorporating substructures into existing models, and (3) a benchmark suite of 13 tasks probing residue-, substructure-, and protein-level representations.

Using Magneton, we develop substructure-tuning, a supervised finetuning method that distills substructural knowledge into pretrained protein models. Across state-of-the-art sequence- and structure-based models, substructure-tuning improves function-related tasks while revealing that substructural signals are complementary to global structural information.
## Contents

- [Magneton](#magneton)
  - [Contents](#contents)
  - [Quickstart](#quickstart)
    - [Installation](#installation)
    - [Downloading datasets](#downloading-datasets)
    - [Finetuning a model](#finetuning-a-model)
    - [Running downstream evaluations](#running-downstream-evaluations)
  - [Datasets](#datasets)
  - [Tasks](#tasks)
  - [Substructure-Tuning](#substructure-tuning)
  - [Evaluations](#evaluations)
  - [Citing](#citing)


## Quickstart
### Installation

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

<!-- Pretrained model weights
Finetuned weights for some combinations (abc) -->
### Downloading datasets
@robert a few commands ideally

### Finetuning a model

Finetuning configurations use Hydra for configuration management via `magneton.cli`. All finetuning scripts are bash files with SLURM headers for GPU job submission. Each finetuning directory contains a `run.sh` script that specifies:

- **Embedding model**: `esm2_150m`, `esm2_650m`, `esm2_3b`, `esmc_300m`, `esmc_600m`, ...
- **Substructure types**: One or more from `Active_site`, `Binding_site`, `Conserved_site`, `Domain`, `Homology`, `Secondary_structure`
- **Output directory**: Automatically set to the current working directory

The following example shows substructure-tuning ESM-C 600M to active site, binding site, and conserved site annotations using EWC:

```bash
cd finetuning/esmc_600m_active_binding_conserved_ewc
sbatch run.sh
```

#### Parameters

- `embedding.model_params.use_flash_attn=True`: Enable Flash Attention (requires optional installation, see above)
- `model.frozen_embedder=False`: Allow backbone finetuning
- `training.loss_strategy=ewc`: Use Elastic Weight Consolidation to prevent catastrophic forgetting
- `training.ewc_weight=400`: EWC regularization weight
- `training.embedding_learning_rate=1e-5`: Learning rate for the embedding model
- `data.substruct_types`: List of substructure types to train on

Upon completion, a checkpoint will be saved as `model_{run_name}.pt` containing the finetuned model.
### Running downstream evaluations

After finetuning, you can evaluate models on the benchmark suite. We automatically prune the substructure-classification MLP head and extract the embedder from saved checkpoints. Evaluation scripts are organized into two directories:

- **`task_specific_no_ft/`**: Evaluations with frozen embeddings (linear probing)
- **`task_specific_ft/`**: Evaluations with task-specific finetuning of the embedding model

Each directory contains subdirectories for different model configurations:
- `{model}_baseline`: Pretrained model without substructure-tuning
- `{model}_abc`: Model substructure-tuned on Active site, Binding site, and Conserved site

#### Example without task-specific finetuning

```bash
cd downstream_evals/task_specific_no_ft/esmc_600m_abc
sbatch run.sh
```

This runs evaluations with a frozen embedding model. Key parameters:
- `model.frozen_embedder=true`: Freeze the embedding model
- `evaluate.model_checkpoint`: Path to the finetuned checkpoint
- `evaluate.has_fisher_info=True`: For checkpoints trained with EWC

#### Example with task-specific finetuning

```bash
cd downstream_evals/task_specific_ft/esmc_600m_abc
sbatch run.sh
```

This allows the embedding model to be finetuned on downstream tasks. Key parameters:
- `model.frozen_embedder=false`: Allow embedding model finetuning
- `training.learning_rate="1e-2"`: Head learning rate
- `training.embedding_learning_rate="2e-5"`: Embedding model learning rate
- `training.max_epochs=20`: Maximum training epochs

#### Evaluation Suites

<!-- Evaluations are run using preset configurations that group related tasks:

- **`deepfri`**: Gene Ontology (GO:MF, GO:BP, GO:CC) and Enzyme Commission (EC) prediction
- **`peer`**: PEER benchmark tasks including GB1, β-lactamase, fluorescence, stability, thermostability, localization, and AAV
- **`proteingym`**: ProteinGym variant effect prediction (zero-shot) -->

Individual task lists can be customized via `evaluate.tasks`. For example:
```bash
evaluate.tasks="['FLIP_bind','human_ppi','biolip_binding','biolip_catalytic']"
```

After all evaluations complete, metrics are automatically aggregated into `combined_metrics.json` in the output directory.

### Using magneton modules functionally
<!-- @robert what do you think? -->

## Datasets
<!-- @robert
Link to data (zenodo?)
Script to download
Table with source of data and how we aggregated annotations
Probably provide a notebook to show all the stats and split creation -->

## Tasks

| Scale | Task | Task type | Metric | Data source | Codebase name |
|-------|------|-----------|--------|-------------|---------------|
| **Interaction** | Human PPI prediction | Binary | Accuracy | [Pan et al. (2010)](https://pubs.acs.org/doi/abs/10.1021/pr100618t) | `human_ppi` |
| **Protein** | Gene Ontology prediction | Multilabel | F<sub>max</sub> | [Gligorijević et al. (2021)](https://www.nature.com/articles/s41467-021-23303-9) | `GO:BP`, `GO:CC`, `GO:MF` |
| | Enzyme Commission prediction | Multilabel | F<sub>max</sub> | [Gligorijević et al. (2021)](https://www.nature.com/articles/s41467-021-23303-9) | `EC` |
| | Subcellular localization | Multiclass | Accuracy | [Almagro Armenteros et al. (2017)](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857) | `saprot_subloc` |
| | Binary localization | Binary | Accuracy | [Almagro Armenteros et al. (2017)](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857) | `binary_localization`, `saprot_binloc` |
| | Thermostability prediction | Regression | Spearman's ρ | [Rao et al. (2019)](https://arxiv.org/abs/1906.08230) | `thermostability`, `saprot_thermostability` |
<!-- | **Substructure** | Substructure classification | Multiclass | Macro accuracy | Ours | `TBD` | -->
| **Residue** | Contact prediction | Binary | Precision@L | [Rao et al. (2019)](https://arxiv.org/abs/1906.08230) | `contact_prediction` |
| | Variant effect prediction | Regression | Spearman's ρ | [Notin et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1) | `proteingym` |
| | Binding residue categorization | Multilabel | F<sub>max</sub> | [Dallago et al. (2021)](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1) | `FLIP_bind` |
| | Functional site prediction | Binary | AUROC | [Yuan et al. (2025)](https://arxiv.org/abs/2503.00089) | `biolip_binding`, `biolip_catalytic` |

<!-- How someone can add an evaluation and what space of evaluations we can support -->

## Substructure-Tuning

Substructure-tuning is a supervised finetuning method that distills substructural knowledge into pretrained protein models. We provide configurations for all model combinations reported in Table 4 in our paper.

### Available Configurations

All finetuning configurations are located in the `finetuning/` directory and organized by model and substructure combination. The naming convention is:

```
{model}_{substruct_types}_{training_strategy}
```

For example:
- `esmc_300m_active_binding_conserved_ewc`: ESM-C 300M trained on Active site, Binding site, and Conserved site with EWC
- `esm2_650m_domain_ewc`: ESM2 650M trained on Domain annotations with EWC

### Supported Models

| Model | Size | Type | Flash Attention |
|-------|------|------|-----------------|
| ESM2 | 150M, 650M, 3B | Sequence | Optional |
| ESM-C | 300M, 600M | Sequence + Structure | Optional |

### Substructure Types

Available substructure annotations from InterPro and SwissProt:

- **Active_site**: Catalytic residues
- **Binding_site**: Ligand/substrate binding residues
- **Conserved_site**: Evolutionarily conserved positions
- **Domain**: Protein domain boundaries
- **Homology**: Homologous superfamily regions
- **Secondary_structure**: Alpha helix, beta strand annotations

### Pretrained Checkpoints

We provide finetuned checkpoints for the substructure combinations reported in our paper at: **[TBD]**

## Evaluations

We provide all evaluation scripts used to generate the results reported in Tables 5 and 6 of our paper. Evaluation configurations are located in `downstream_evals/` and can be run as described in the Quickstart section.

### Evaluation Modes

**Task-Specific No Finetuning (`task_specific_no_ft/`)**: Linear probing with frozen embeddings
- Trains only a task-specific head on top of frozen representations
- Used to assess quality of pretrained/substructure-tuned representations

**Task-Specific Finetuning (`task_specific_ft/`)**: Full model finetuning
- Finetunes both the embedding model and task head on downstream tasks
- Shows how well representations adapt to new tasks

**Note on batch sizes**: Larger models (ESM2 650M, 3B) use a lower batch size (`data.batch_size`) to fit in GPU memory, with `training.accumulate_grad_batches` adjusted to maintain effective batch size.

## Citing
Arxiv link: TBA

Please consider citing `magneton` if it proves useful in your work.

```bibtex
TBA

```
