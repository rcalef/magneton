import torch
import pandas as pd

from tqdm import tqdm

from magneton.config import PipelineConfig
from magneton.training.embedding_mlp import EmbeddingMLP

def classify_substructs(
    model: EmbeddingMLP,
    data: torch.utils.data.DataLoader,
    config: PipelineConfig,
):
    # UMAP
    # Get embeddings for all substructures
    all_embeddings = []

    results = {
        'protein_ids': [],
        'true_labels': [],
        'true_mappings': [],
        'pred_labels': [],  # Store argmax predictions
        'pred_mappings': [],
        'top_k_labels': [],  # Store top-k predictions
        'top_k_probs': []   # Store top-k probabilities
    }
    accs = []
    k = 3

    # TODO Change based on type: Domain tsv
    labels_tsv_path = f'/weka/scratch/weka/kellislab/rcalef/data/interpro/103.0/label_sets/selected_subset/{list(config.data.substruct_types)[0]}.labels.tsv'
    labels_df = pd.read_csv(labels_tsv_path, sep='\t')
    label_to_element = dict(zip(labels_df['label'], labels_df['element_name']))

    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating batches"):
            # Skip batch if no substructs
            # TODO Reevaluate how to deal with batches that have only proteins with no substructures
            if [] in batch.substructures:
                continue

            # Get substructure embeddings
            substruct_embeds = model.embed(batch)
            all_embeddings.append(substruct_embeds)

            # Get protein id
            # batch_ids = batch.prot_ids
            batch_ids = [batch.prot_ids[i] for i, prot_substructs in enumerate(batch.substructures)
                    for substruct in prot_substructs]

            # Get labels
            batch_labels = [substruct.label for prot_substructs in batch.substructures
                        for substruct in prot_substructs]

            # Forward pass
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Map from id to class
            true_mappings = [label_to_element.get(label, f"Unknown label: {label}") for label in batch_labels]
            pred_mappings = [label_to_element.get(label, f"Unknown label: {label}") for label in [int(x.item()) for x in preds]]

            # Logging metrics
            results['protein_ids'].extend(batch_ids)
            results['true_labels'].extend(batch_labels)
            results['true_mappings'].extend(true_mappings)

            # results['pred_probs'].extend([str(x) for x in probs])
            results['pred_labels'].extend([int(x.item()) for x in preds])
            results['pred_mappings'].extend(pred_mappings)

            results['top_k_labels'].extend(topk_indices.cpu().numpy().tolist())
            results['top_k_probs'].extend(topk_probs.cpu().numpy().tolist())

            acc = model.train_acc(preds, torch.tensor(batch_labels, device=logits.device))
            accs.append(acc)
            print(acc)

    # Combine embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Write results
    df = pd.DataFrame(results)

    # TODO Change based on type
    csv_path = config.output_dir / f"evaluation_{config.run_id}_{list(config.data.substruct_types)[0]}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Accuracies for each batch: {accs}")
    print(f"Saved detailed evaluation results to {config.output_dir}")