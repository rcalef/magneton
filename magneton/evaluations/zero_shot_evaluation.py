import os
import re
import attr
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ESM imports
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from magneton.config import PipelineConfig


def run_zero_shot_evaluation(
    config: PipelineConfig,
    task: str,
    output_dir: str,
    run_id: str,
) -> Dict[str, Any]:
    """
    Run zero-shot evaluation on ProteinGym DMS benchmark using a lean implementation.

    Args:
        config: Pipeline configuration
        task: Task name (should be "zero_shot")
        output_dir: Output directory for results
        run_id: Unique run identifier

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Running zero-shot evaluation for {task}")
    print(f"Output directory: {output_dir}")
    print(f"Run ID: {run_id}")

    # Validate that we have the required configuration
    if not hasattr(config.evaluate, "model_checkpoint"):
        raise ValueError("model_checkpoint must be specified in evaluation config")

    model_checkpoint = config.evaluate.model_checkpoint
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    # Set up paths
    proteingym_base = (
        "/net/vast-storage/scratch/vast/kellislab/artliang/magneton/external/proteingym"
    )
    dms_data_folder = f"{proteingym_base}/DMS_ProteinGym_substitutions"
    dms_reference_file = f"{proteingym_base}/reference_files/DMS_substitutions.csv"

    if not os.path.exists(dms_reference_file):
        raise FileNotFoundError(f"DMS reference file not found: {dms_reference_file}")

    # Create output directories
    run_output_dir = os.path.join(output_dir, run_id)
    scores_output_dir = os.path.join(run_output_dir, "scores")
    os.makedirs(scores_output_dir, exist_ok=True)

    # Load model once
    print(f"Loading model from: {model_checkpoint}")
    model = load_esmc_model(model_checkpoint)

    # Load DMS reference file
    print(f"Loading DMS reference from: {dms_reference_file}")
    reference_df = pd.read_csv(dms_reference_file)

    # Score all assays
    print(f"Processing {len(reference_df)} DMS assays...")
    all_results = {}

    for idx, row in tqdm(
        reference_df.iterrows(), total=len(reference_df), desc="Processing assays"
    ):
        dms_id = row["DMS_id"]
        target_seq = row["target_seq"]

        try:
            # Load DMS data
            dms_file_path = os.path.join(dms_data_folder, f"{dms_id}.csv")
            if not os.path.exists(dms_file_path):
                print(f"Warning: DMS file not found for {dms_id}")
                continue

            dms_df = pd.read_csv(dms_file_path)

            # Score mutations for this assay
            mutations = dms_df["mutant"].tolist()
            mutation_scores = score_mutations_lean(
                sequence=target_seq,
                mutations=mutations,
                model=model,
                model_type="esmc_300M",
            )

            # Add scores to dataframe and save
            score_column = f"{run_id}_score"
            dms_df[score_column] = dms_df["mutant"].map(
                lambda x: mutation_scores.get(x, np.nan)
            )

            # Save individual assay results
            output_file = os.path.join(scores_output_dir, f"{dms_id}.csv")
            dms_df.to_csv(output_file, index=False)

            # Calculate basic metrics for this assay
            assay_metrics = calculate_assay_metrics(dms_df, "DMS_score", score_column)
            all_results[dms_id] = assay_metrics

            print(
                f"Processed {dms_id}: Spearman = {assay_metrics.get('spearman', np.nan):.4f}"
            )

        except Exception as e:
            print(f"Error processing {dms_id}: {str(e)}")
            all_results[dms_id] = {"error": str(e)}
            continue

    # Calculate aggregate performance metrics
    performance_summary = calculate_performance_summary(
        all_results, reference_df, run_id
    )

    # Save results
    results_file = os.path.join(run_output_dir, f"performance_summary_{run_id}.json")
    import json

    with open(results_file, "w") as f:
        json.dump(performance_summary, f, indent=2, default=str)

    print(f"Zero-shot evaluation completed for {run_id}")
    print(
        f"Mean Spearman correlation: {performance_summary.get('mean_spearman', np.nan):.4f}"
    )
    return performance_summary


def load_esmc_model(model_path: str) -> ESMC:
    """Load ESMC model from checkpoint."""
    print(f"Loading ESMC model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cuda", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load base model and apply custom weights
    model = ESMC.from_pretrained("esmc_300m")
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")
    model.eval()

    return model


def score_mutations_lean(
    sequence: str,
    mutations: List[str],
    model: ESMC,
    model_type: str = "esmc_300M",
    window_size: int = 1024,
) -> Dict[str, float]:
    """
    Lean version of mutation scoring using masked-marginals approach.

    Args:
        sequence: Protein sequence
        mutations: List of mutations in format "A25G"
        model: ESMC model instance
        model_type: Model type identifier
        window_size: Window size for long sequences

    Returns:
        Dictionary of mutation scores
    """
    if len(sequence) == 0:
        raise ValueError("Empty sequence provided")

    if not mutations:
        return {}

    print(f"Scoring {len(mutations)} mutations on sequence of length {len(sequence)}")

    # Parse mutations
    parsed_mutations = []
    for mutation in mutations:
        parsed_mut = parse_mutation(mutation, sequence)
        if parsed_mut is not None:
            parsed_mutations.append(parsed_mut)

    if not parsed_mutations:
        print("No valid mutations to score")
        return {}

    # Build amino acid to token mapping
    aa_to_token = build_aa_token_mapping(model)

    # Create protein object
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    sequence_tokens = protein_tensor.sequence

    # Get all unique positions that need scoring
    positions_to_score = get_unique_positions(parsed_mutations)

    # Check if windowing is needed
    seq_len = len(sequence)
    needs_windowing = seq_len > window_size - 2

    if needs_windowing:
        print(f"Using windowing for long sequence (length {seq_len})")

    # Score each position
    position_probs = score_positions(
        protein,
        protein_tensor,
        positions_to_score,
        model,
        needs_windowing,
        window_size,
        sequence,
    )

    # Calculate mutation scores
    mutation_scores = {}
    for wt, pos_list, mt, seq_pos_list, mutation_name in parsed_mutations:
        try:
            score = calculate_mutation_score(
                wt, mt, seq_pos_list, position_probs, aa_to_token
            )
            mutation_scores[mutation_name] = score
        except Exception as e:
            print(f"Error scoring {mutation_name}: {e}")
            mutation_scores[mutation_name] = np.nan

    return mutation_scores


def parse_mutation(mutation: str, sequence: str) -> Optional[Tuple]:
    """Parse a mutation string and validate against sequence."""
    if ":" in mutation:
        # Handle multiple mutations
        sub_mutations = mutation.split(":")
        multi_wt, multi_mt = "", ""
        multi_pos, multi_seq_pos = [], []

        for sub_mut in sub_mutations:
            match = re.match(r"([A-Z])(\d+)([A-Z])", sub_mut)
            if not match:
                return None

            wt, pos_str, mt = match.groups()
            pos = int(pos_str)
            seq_pos = pos - 1  # Convert to 0-indexed

            # Validate
            if seq_pos < 0 or seq_pos >= len(sequence):
                return None
            if sequence[seq_pos] != wt:
                return None

            multi_wt += wt
            multi_mt += mt
            multi_pos.append(pos)
            multi_seq_pos.append(seq_pos)

        return (multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation)

    else:
        # Handle single mutation
        match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
        if not match:
            return None

        wt, pos_str, mt = match.groups()
        pos = int(pos_str)
        seq_pos = pos - 1

        # Validate
        if seq_pos < 0 or seq_pos >= len(sequence):
            return None
        if sequence[seq_pos] != wt:
            return None

        return (wt, [pos], mt, [seq_pos], mutation)


def build_aa_token_mapping(model: ESMC) -> Dict[str, int]:
    """Build mapping from amino acids to token IDs."""
    aa_to_token = {}
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    for aa in amino_acids:
        encoded_protein = ESMProtein(sequence=aa)
        tensor = model.encode(encoded_protein)
        aa_to_token[aa] = tensor.sequence[1].item()  # Skip BOS token

    return aa_to_token


def get_unique_positions(parsed_mutations: List[Tuple]) -> List[int]:
    """Get all unique sequence positions that need to be scored."""
    all_positions = []
    for _, _, _, seq_positions, _ in parsed_mutations:
        if isinstance(seq_positions, list):
            all_positions.extend(seq_positions)
        else:
            all_positions.append(seq_positions)
    return sorted(set(all_positions))


def score_positions(
    protein: ESMProtein,
    protein_tensor,
    positions_to_score: List[int],
    model: ESMC,
    needs_windowing: bool,
    window_size: int,
    sequence: str,
) -> Dict[int, torch.Tensor]:
    """Score all positions using masked language modeling."""
    position_probs = {}
    mask_token_id = getattr(model, "mask_token_id", 32)

    for seq_pos in tqdm(positions_to_score, desc="Scoring positions", leave=False):
        token_pos = seq_pos + 1  # Add 1 for BOS token

        if needs_windowing:
            # Handle long sequences with windowing
            window_half = (window_size - 2) // 2
            start_pos = max(0, seq_pos - window_half)
            end_pos = min(len(sequence), start_pos + window_size - 2)

            if end_pos == len(sequence):
                start_pos = max(0, len(sequence) - (window_size - 2))

            # Create windowed protein
            window_seq = sequence[start_pos:end_pos]
            window_protein = ESMProtein(sequence=window_seq)
            window_tensor = model.encode(window_protein)

            # Calculate position in window
            window_token_pos = seq_pos - start_pos + 1

            # Apply mask
            masked_tokens = window_tensor.sequence.clone()
            masked_tokens[window_token_pos] = mask_token_id
            masked_tensor = attr.evolve(window_tensor, sequence=masked_tokens)

            scoring_pos = window_token_pos
        else:
            # Use full sequence
            masked_tokens = protein_tensor.sequence.clone()
            masked_tokens[token_pos] = mask_token_id
            masked_tensor = attr.evolve(protein_tensor, sequence=masked_tokens)

            scoring_pos = token_pos

        # Get logits
        with torch.no_grad():
            logits_output = model.logits(masked_tensor, LogitsConfig(sequence=True))

        # Calculate probabilities
        token_logits = logits_output.logits.sequence[0, scoring_pos]
        token_probs = torch.log_softmax(token_logits, dim=-1)
        position_probs[seq_pos] = token_probs

    return position_probs


def calculate_mutation_score(
    wt: str,
    mt: str,
    seq_pos_list: List[int],
    position_probs: Dict[int, torch.Tensor],
    aa_to_token: Dict[str, int],
) -> float:
    """Calculate mutation score from position probabilities."""
    score = 0.0

    if len(seq_pos_list) > 1:
        # Multiple mutation
        for i, seq_pos in enumerate(seq_pos_list):
            single_wt = wt[i]
            single_mt = mt[i]

            wt_idx = aa_to_token[single_wt]
            mt_idx = aa_to_token[single_mt]

            token_probs = position_probs[seq_pos]
            score += (token_probs[mt_idx] - token_probs[wt_idx]).item()
    else:
        # Single mutation
        seq_pos = seq_pos_list[0] if isinstance(seq_pos_list, list) else seq_pos_list

        wt_idx = aa_to_token[wt]
        mt_idx = aa_to_token[mt]

        token_probs = position_probs[seq_pos]
        score = (token_probs[mt_idx] - token_probs[wt_idx]).item()

    return score


def calculate_assay_metrics(
    df: pd.DataFrame, true_col: str, pred_col: str
) -> Dict[str, float]:
    """Calculate performance metrics for a single assay."""
    # Drop missing values
    valid_data = df.dropna(subset=[true_col, pred_col])

    if len(valid_data) == 0:
        return {"num_valid": 0, "error": "no_valid_data"}

    y_true = valid_data[true_col].values
    y_pred = valid_data[pred_col].values

    metrics = {"num_valid": len(valid_data)}

    try:
        # Spearman correlation
        metrics["spearman"] = spearmanr(y_true, y_pred)[0]
    except:
        metrics["spearman"] = np.nan

    try:
        # NDCG
        metrics["ndcg"] = calc_ndcg_lean(y_true, y_pred)
    except:
        metrics["ndcg"] = np.nan

    try:
        # Top recall
        metrics["top_recall"] = calc_toprecall_lean(y_true, y_pred)
    except:
        metrics["top_recall"] = np.nan

    try:
        # AUC (requires binary labels)
        if "DMS_score_bin" in valid_data.columns:
            y_true_bin = valid_data["DMS_score_bin"].values
            metrics["auc"] = roc_auc_score(y_true_bin, y_pred)
        else:
            metrics["auc"] = np.nan
    except:
        metrics["auc"] = np.nan

    try:
        # MCC
        if "DMS_score_bin" in valid_data.columns:
            y_true_bin = valid_data["DMS_score_bin"].values
            median_cutoff = np.median(y_pred)
            y_pred_bin = (y_pred >= median_cutoff).astype(int)
            metrics["mcc"] = matthews_corrcoef(y_true_bin, y_pred_bin)
        else:
            metrics["mcc"] = np.nan
    except:
        metrics["mcc"] = np.nan

    return metrics


def calc_ndcg_lean(
    y_true: np.ndarray, y_score: np.ndarray, top_percent: int = 10
) -> float:
    """Lean implementation of NDCG calculation."""
    k = np.floor(y_true.shape[0] * (top_percent / 100)).astype(int)

    # Min-max normalization for gains
    gains = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true))
    ranks = np.argsort(np.argsort(-y_score)) + 1

    # Top k elements
    ranks_k = ranks[ranks <= k]
    gains_k = gains[ranks <= k]

    # Filter out zero gains
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]

    if len(ranks_fil) == 0:
        return 0.0

    # DCG calculation
    dcg = np.sum([g / np.log2(r + 1) for r, g in zip(ranks_fil, gains_fil)])

    # IDCG calculation
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= k]
    ideal_gains_k = gains[ideal_ranks <= k]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum(
        [g / np.log2(r + 1) for r, g in zip(ideal_ranks_fil, ideal_gains_fil)]
    )

    return dcg / idcg if idcg > 0 else 0.0


def calc_toprecall_lean(
    true_scores: np.ndarray, model_scores: np.ndarray, top_percent: int = 10
) -> float:
    """Lean implementation of top recall calculation."""
    top_true = true_scores >= np.percentile(true_scores, 100 - top_percent)
    top_model = model_scores >= np.percentile(model_scores, 100 - top_percent)

    tp = np.sum(top_true & top_model)
    return tp / np.sum(top_true) if np.sum(top_true) > 0 else 0.0


def calculate_performance_summary(
    all_results: Dict[str, Dict], reference_df: pd.DataFrame, run_id: str
) -> Dict[str, Any]:
    """Calculate aggregate performance summary across all assays."""
    # Extract successful results
    valid_results = {
        k: v
        for k, v in all_results.items()
        if "error" not in v and "spearman" in v and not np.isnan(v["spearman"])
    }

    if not valid_results:
        return {"error": "no_valid_results", "num_processed": len(all_results)}

    # Calculate aggregate metrics
    spearman_values = [v["spearman"] for v in valid_results.values()]
    ndcg_values = [
        v["ndcg"]
        for v in valid_results.values()
        if "ndcg" in v and not np.isnan(v["ndcg"])
    ]
    auc_values = [
        v["auc"]
        for v in valid_results.values()
        if "auc" in v and not np.isnan(v["auc"])
    ]
    mcc_values = [
        v["mcc"]
        for v in valid_results.values()
        if "mcc" in v and not np.isnan(v["mcc"])
    ]

    summary = {
        "run_id": run_id,
        "num_assays_processed": len(all_results),
        "num_assays_valid": len(valid_results),
        "mean_spearman": np.mean(spearman_values),
        "std_spearman": np.std(spearman_values),
        "mean_ndcg": np.mean(ndcg_values) if ndcg_values else np.nan,
        "mean_auc": np.mean(auc_values) if auc_values else np.nan,
        "mean_mcc": np.mean(mcc_values) if mcc_values else np.nan,
        "valid_assay_ids": list(valid_results.keys()),
        "individual_results": valid_results,
    }

    return summary
