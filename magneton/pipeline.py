from pathlib import Path

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from magneton.config import PipelineConfig
from magneton.data.utils import get_dataloader
from magneton.embedders.factory import EmbedderFactory
from magneton.training.trainer import ModelTrainer
# from .visualization import EmbeddingVisualizer

class EmbeddingPipeline:
    """Main pipeline for protein embedding and analysis"""

    def __init__(self, cfg: PipelineConfig):
        print("\n=== Pipeline Configuration ===")
        print(f"Output Directory: {cfg.pipeline.output_dir}")
        print(f"Model Type: {cfg.pipeline.model.model_type}")
        print(f"Embedding Config: {cfg.pipeline.embedding}")
        print("============================\n")

        self.config = cfg
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embedder = EmbedderFactory.create_embedder(self.config.embedding)
        self.trainer = ModelTrainer(self.config)
        # self.visualizer = EmbeddingVisualizer()

    def run(self):
        """Run complete pipeline"""
        self.run_embedding()
        self.run_training()
        self.run_visualization()

    def run_embedding(self):
        """Generate and save embeddings"""
        print("Generating embeddings...")

        # TODO
        # Check if embeddings exist, terminate if so
        # Implement override if want to regenerate

        # Get data loader
        loader = get_dataloader(
            self.config.data,
            self.embedder.get_required_input_type(),
        )

        # Get embeddings and associated IDs
        all_embeds = []
        all_ids = []
        for batch in tqdm(loader, desc=f"{self.embedder.model_name()} embedding"):
            batch_embeds = self.embedder.embed_batch(batch)
            all_embeds.extend(batch_embeds)
            all_ids.extend([prot.uniprot_id for (seq, prot) in batch])

        # Save results
        embedding_path = self.output_dir / f"{self.embedder.model_name()}.embeddings.pt"
        all_embeds = torch.cat(all_embeds, dim=0).cpu()
        torch.save(all_embeds, embedding_path)

        ids_path = self.output_dir / f"{self.embedder.model_name()}.ids.tsv"
        pd.Series(all_ids).to_csv(ids_path, index=False, header=False, sep="\t")
        print("Done generating embeddings")

    def run_training(self):
        """Train and evaluate model using Lightning"""
        print("Training model...")

        # Need for datamodule?

        # Load saved embeddings
        embeddings = np.load(self.output_dir / "embeddings.npy")
        labels = np.load(self.output_dir / "labels.npy")

        # Train model
        metrics = self.trainer.train_and_evaluate(embeddings, labels)

        # Save model
        model_path = self.output_dir / "model.pt"
        self.trainer.save_model(model_path)
        print(f"Saved model to {model_path}")
        print(f"Training metrics: {metrics}")

        return metrics

    def run_visualization(self):
        """Generate visualizations"""
        print("Generating visualizations...")

        # Load data
        embeddings = np.load(self.output_dir / "embeddings.npy")
        labels = np.load(self.output_dir / "labels.npy")

        # Generate visualizations
        self.visualizer.visualize_embeddings(
            embeddings,
            labels,
            save_dir=self.output_dir
        )
        print(f"Saved visualizations to {self.output_dir}")