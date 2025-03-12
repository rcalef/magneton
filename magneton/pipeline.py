from pathlib import Path
from pprint import pprint

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from magneton.config import PipelineConfig
from magneton.embedders.factory import EmbedderFactory
from magneton.training.trainer import ModelTrainer
from magneton.training.embedding_mlp import EmbeddingMLP
# from .visualization import EmbeddingVisualizer

class EmbeddingPipeline:
    """Main pipeline for protein embedding and analysis"""

    def __init__(self, cfg: PipelineConfig):
        print("\n=== Pipeline Configuration ===")
        print(f"Output Directory: {cfg.output_dir}")
        print(f"Model Type: {cfg.model.model_type}")
        print(f"Full Config:")
        pprint(cfg, compact=False)
        print("============================\n")

        self.config = cfg
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset
        _, _, data_cls  = EmbedderFactory.fetch_embedder_classes(self.config.embedding.model)
        self.data_module = data_cls(self.config.data, self.config.training)

    def run(self):
        """Run complete pipeline"""
        self.run_embedding()
        self.run_training()
        self.run_visualization()

    def run_embedding(self):
        """Generate and save embeddings"""
        print("Generating embeddings...")
        embedder  = EmbedderFactory.create_embedder(self.config.embedding)

        # TODO
        # Check if embeddings exist, terminate if so
        # Implement override if want to regenerate

        # Get data loader
        loader = self.data_module.predict_dataloader()

        # Get embeddings and associated IDs
        all_embeds = []
        for batch in tqdm(loader, desc=f"{embedder.model_name()} embedding"):
            batch = batch.to(self.config.embedding.device)
            batch_embeds = embedder.embed_batch(batch)
            all_embeds.append(batch_embeds.detach().cpu())

        # Save results
        embedding_path = self.output_dir / f"{embedder.model_name()}.embeddings.pt"
        all_embeds = torch.cat(all_embeds, dim=0).cpu()
        torch.save(all_embeds, embedding_path)

        print("Done generating embeddings")

    def run_training(self):
        """Train and evaluate model using Lightning"""
        print("Training model...")
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        # Train model
        model = EmbeddingMLP(
            config=self.config,
            num_classes=len(train_loader.dataset.substruct_parser.type_to_label)
        )
        trainer = ModelTrainer(self.config.training, self.output_dir)
        trainer.setup(model)

        metrics = trainer.train_and_evaluate(train_loader=train_loader, val_loader=val_loader)

        # Save model
        model_path = self.output_dir / "model.pt"
        trainer.save_model(model_path)
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