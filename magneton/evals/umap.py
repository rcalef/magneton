import torch
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

class UMAPVisualizer:
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1):
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        save_dir: Optional[Path] = None,
        title: str = "UMAP of Protein Substructure Embeddings",
        type: str = None,
    ):
        """Create UMAP visualization of embeddings"""
        # Convert to numpy and standardize
        X = embeddings.cpu().numpy()
        X = StandardScaler().fit_transform(X)
        
        # Fit and transform with UMAP
        embedding_2d = self.reducer.fit_transform(X)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if labels:
            # Create scatter plot with labels
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = np.array(labels) == label
                plt.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    label=label,
                    alpha=0.6
                )
            plt.legend()
        else:
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6)
            
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"umap_visualization_{type}.png")
        plt.close()