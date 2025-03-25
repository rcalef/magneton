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
from magneton.evals.umap import UMAPVisualizer

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

        self.test_dir = Path(self.config.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset
        _, _, data_cls  = EmbedderFactory.fetch_embedder_classes(self.config.embedding.model)
        self.data_module = data_cls(self.config.data, self.config.training)

    def run(self):
        """Run complete pipeline"""
        self.run_embedding()
        self.run_training()
        self.run_evals()

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

    def run_evals(self):
        """Generate visualizations and evals"""
        print("Evaluating and generating visualizations...")

        # Load model and data
        # Sample loading in ckpt for esmc embedding mlp
        # TODO Set up flag for this in the config file
        model = EmbeddingMLP.load_from_checkpoint(
            "/net/vast-storage/scratch/vast/kellislab/artliang/magneton/magneton/runs/kcmou838/checkpoints/epoch=2-val_f1=0.99.ckpt",
        )
        model.eval()

        # TODO
        # Generate evals

        # UMAP
        # Get embeddings for all substructures
        all_embeddings = []
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for batch in self.data_module.test_dataloader():
                # Skip batch if no substructs
                # print(batch.substructures)
                if [] in batch.substructures:
                    continue

                # Get substructure embeddings
                substruct_embeds = model.embed(batch)
                all_embeddings.append(substruct_embeds)
                
                # Get labels
                batch_labels = [substruct.label for prot_substructs in batch.substructures 
                            for substruct in prot_substructs]
                true_labels.extend(batch_labels)

                logits = model(batch)
                pred = torch.argmax(logits, dim=1)
                pred_labels.extend([str(x.item()) for x in pred])
        
        # Combine embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Create UMAP visualization
        visualizer = UMAPVisualizer()
        visualizer.visualize_embeddings(
            all_embeddings,
            pred_labels,
            save_dir=self.test_dir / "visualizations",
            title=f"UMAP of {self.config.embedding.model} Substructure Embeddings With Predicted Labels",
            type="pred",
        )
        visualizer.visualize_embeddings(
            all_embeddings,
            true_labels,
            save_dir=self.test_dir / "visualizations",
            title=f"UMAP of {self.config.embedding.model} Substructure Embeddings",
            type="true",
        )

        print(f"Saved evals to {self.output_dir}")