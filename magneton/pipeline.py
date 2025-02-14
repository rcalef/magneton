from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from omegaconf import DictConfig
from .embedders.factory import EmbedderFactory
# TODO Fix this with Robert's data entry
from magneton.io.internal import ProteinDataset
from .training.trainer import ModelTrainer
# from .visualization import EmbeddingVisualizer

class EmbeddingPipeline:
    """Main pipeline for protein embedding and analysis"""
    
    def __init__(self, cfg: DictConfig):
        print("\n=== Pipeline Configuration ===")
        print(f"Output Directory: {cfg.pipeline.output_dir}")
        print(f"Model Type: {cfg.pipeline.model.model_type}")
        print(f"Embedding Config: {cfg.pipeline.embedding}")
        print("============================\n")
        
        self.config = cfg.pipeline
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedder = EmbedderFactory.create_embedder(self.config.embedding)
        self.dataset = ProteinDataset(self.config.data.data_dir)
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

        # Load data
        proteins = self.dataset.load_proteins()
        
        # Generate embeddings
        # Make this function model-agnostic because different embedders require different types of data (seq vs. structure)
        embeddings = self.embedder.embed_batch(proteins)
        labels = self.dataset.get_labels(proteins)
        
        # Save results
        embedding_path = self.output_dir / "embeddings.npy"
        labels_path = self.output_dir / "labels.npy"
        np.save(embedding_path, embeddings)
        np.save(labels_path, labels)
        print(f"Saved embeddings to {embedding_path}")
        
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