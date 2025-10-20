# Magneton architecture

The structure of Magneton was intended to make it easy to add new models and evaluation tasks, where the new models could either be integration of external base protein models (similar to what has been done with ESM, SaProt, ProSST), completely novel base protein models, or head models for new evaluation tasks.

## Organization
The high-level organization of the codebase is as follows:
  - `core_types.py` - Objects for representing proteins and their associated substructures. These are the types used in the Magneton dataset JSONL files in our HuggingFace dataset.
  - `config.py` - Objects used for configuring Magneton runs using Hydra.
  - `cli.py` - Main entrypoint script.
  - `pipeline.py` - Runners for model training and evaluation.
  - `data/` - Datasets for model training and evaluation. `data_modules.py` provides the main user-facing entrypoint via Lightning `DataModules`.
    - `core/` - Datasets and types for the Magneton dataset consisting of proteins with annotated substructures and, optionally, amino acid sequences and paths to structure files.
    - `evaluations/` - Datasets for evaluation tasks.
    - `model_specific/` - Transformations that operate on top of the datasets defined in `core` and `evaluations` to produce model-specific inputs
  - `evaluations/` - Evaluation loops and metrics for different downstream evaluation tasks.
  - `io/` - Functions for interacting with the Magneton JSONL datasets as well as raw InterPro XML files.
  - `models/` - Model implementations for substructure classification, tuning, and downstream evaluations.
    - `evaluation_classifier.py` - Overarching model used for downstream evaluations. Contains both a base model and a classification head.
    - `substructure_classifier.py` - Overarching model used for substructure classification and tuning. Contains a base model.
    - `base_models/` - Base model implementations (e.g. ESM-C, ProSST) which, given a batch of proteins, produce per-residue or per-protein embeddings.
    - `head_classifiers/` - Head models used for various downstream evaluation tasks (e.g. protein-level tasks, residue-level tasks).
  - `training/` - Trainer used for substructure-tuning and supervised downstream evaluation tasks.

## Adding a new model
Adding a new model just takes three steps:
1. Define the model-specific data transformations
2. Implement the forward pass which produces residue- or protein-level embeddings given an input batch
3. Register your model

### Data transform
The data transforms are implemented as a node from [torchdata.nodes](https://meta-pytorch.org/data/main/what_is_torchdata_nodes.html). The transform should take in a `DataElement` object as defined [here](./data/core/unified_dataset.py#24) and return a model-specific data element with the appropriate transformations applied. The transform should also define how to collate multiple data elements into a batch.

[ESM-C](./data/model_specific/esmc.py#L36) is a simple example, where the transformation just tokenizes the sequences in the received data. While the actual code is a bit more complex to handle more use cases, here's a simplified demonstration:
```python
# Copied over for clarity.
@dataclass(kw_only=True)
class DataElement:
    """Single dataset entry."""

    protein_id: str
    length: int
    seq: str | None = None

@dataclass(kw_only=True)
class Batch:
  """Batch of protein data."""

    protein_ids: list[str]
    lengths: list[int]
    seqs: list[str] | None = None

# Code specifically for ESM-C
@dataclass(kw_only=True)
class ESMCDataElement(DataElement):
    """Single data element for ESM-C."""

    tokenized_seq: torch.Tensor

@dataclass(kw_only=True)
class ESMCBatch(Batch):
    """Batch of data for ESM-C."""

    tokenized_seq: torch.Tensor

class ESMCTransformNode(ParallelMapper):
    def __init__(
        self,
        source_node: BaseNode,
        num_workers: int = 2,
    ):
        tokenizer = get_esmc_model_tokenizers()

        def _process(
            x: DataElement,
        ) -> ESMCDataElement:
            return ESMCDataElement(
                protein_id=x.protein_id,
                length=x.length,
                tokenized_seq=torch.tensor(tokenizer.encode(x.seq)),
            )

        super().__init__(source=source_node, map_fn=_process, num_workers=num_workers)

    def get_collate_fn(
        self,
    ) -> Callable:
        tokenizer = get_esmc_model_tokenizers()
        return partial(
            esmc_collate,
            pad_id=tokenizer.pad_token_id,
        )

def esmc_collate(
    entries: list[ESMCDataElement],
    pad_id: int,
) -> ESMCBatch:
    """
    Collate the entries into a batch.
    """
    protein_ids = [x.protein_id for x in entries]
    lengths = [x.length for x in entries]
    padded_tokens = stack_variable_length_tensors(
        [x.tokenized_seq for x in entries],
        constant_value=pad_id,
    )
    return ESMCBatch(
        protein_ids=protein_ids,
        lengths=lengths,
        tokenized_seq=padded_tokens,
    )
```

A more complex example is something like SaProt, where we also need to generate structural tokens. In this case, the transform node can also implement logic like calculating structure tokens on the fly given a PDB path, reading precomputed structure tokens from a file, or the best of both worlds: computing and saving structure tokens if not present, and reading the precomputed tokens if so. Since this is a bit more involved, we'll just refer to [the implementation](./data/model_specific/saprot.py#L182) rather than pasting a snippet here.

### Base model implementation
The second part of adding a new model is adding the base model implementation. All base models follow the abstract `BaseModel` interface defined [here](./models/base_models/interface.py#L18) in `models/base_models/interface.py`. At its core, the base model just takes in the model-specific data batch defined above and outputs embeddings. The overarching models for substructure-tuning and downstream evaluation handle the rest of the logic. To train your model with EWC, you'll also have to implement the calculation of the model's original loss given a batch of data (e.g. MLM loss for ESM-C).

### Registering your implementations
Finally, you just need to register your dataset implementation in [`data/data_modules.py`](./data/data_modules.py#L51) and the base model implementation in [`models/base_models/factory.py`](./models/base_models/factory.py#L11). That's it! You can now use your model for substructure classification, substructure-tuning, and all the evaluation tasks implemented in Magneton!

## Adding a new evaluation task
Adding a new evaluation task is similarly easy and again takes three steps:
  1. Add your dataset such that it returns elements as the [`DataElement` class](./data/core/unified_dataset.py#L24). Model-specific transforms will automatically be applied to your dataset. For examples, please see the other [evaluation dataset implementations](./data/evaluations/).
  2. [Register your task's type](./data/evaluations/task_types.py#L18) (and give it a name!). This ensures that the appropriate type of loss function and metrics are used. We currently support multilabel, multiclass, binary, and regression tasks. Please reach out if these don't meet your need.
  3. [Register you task's granularity](./data/data_modules.py#L214). This ensures the appropriate type of head classifier is used. We currently support protein- and residue-level tasks, contact prediction, and PPI (i.e. protein pair) tasks. Again, please reach out if these don't meet your need.

That's it! Your task should now be usable with all of the models implemented in Magneton. 