from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from .embedders.factory import EmbedderFactory
from .config.base_config import ESMConfig, GearNetConfig
from .dataset import ProteinDataset
from magneton.io.internal import ProteinDataset
# dataset deprecated
# train, visualize, mlp are also deprecated
from .training.trainer import ModelTrainer
from .visualization import EmbeddingVisualizer
from .config.config import Config
from .data_module import ProteinDataModule

class EmbeddingPipeline:
    """Main pipeline for protein embedding and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        # Convert dict config to Config object for training
        self.config = Config(**config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedder = EmbedderFactory.create_embedder(self.config)
        self.dataset = ProteinDataset(self.config.data)
        self.trainer = ModelTrainer(self.config)
        self.visualizer = EmbeddingVisualizer(self.config)
        
    def _create_config(self, config_dict: Dict[str, Any]):
        """Create appropriate config object based on model type"""
        model_type = config_dict.get('model_type')
        if model_type == 'esm':
            return ESMConfig(**config_dict)
        elif model_type == 'gearnet':
            return GearNetConfig(**config_dict)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def run(self):
        """Run complete pipeline"""
        self.run_embedding()
        self.run_training()
        self.run_visualization()
        
    def run_embedding(self):
        """Generate and save embeddings"""
        print("Generating embeddings...")
        
        # Load data
        proteins = self.dataset.load_proteins(self.config.num_proteins)
        
        # Generate embeddings
        embeddings = self.embedder.embed_sequences(proteins)
        labels = self.dataset.get_labels(proteins)
        
        # Save results
        np.save(self.output_dir / f"{self.config.data.embedding_type}_embeddings.npy", embeddings)
        np.save(self.output_dir / f"{self.config.data.embedding_type}_labels.npy", labels)
        
    def run_training(self):
        """Train and evaluate model using Lightning"""
        print("Training model...")
        
        # Create data module
        datamodule = ProteinDataModule(self.config.data)
        datamodule.setup()
        
        # Setup trainer with dataset info
        self.trainer.setup(
            num_classes=datamodule.num_classes,
            input_dim=datamodule.input_dim
        )
        
        # Train and evaluate
        metrics = self.trainer.train_and_evaluate(datamodule)
        
        # Save model
        self.trainer.save_model(
            self.output_dir / f"{self.config.data.embedding_type}_model.pt"
        )
        
        return metrics
        
    def run_visualization(self):
        """Generate visualizations"""
        print("Generating visualizations...")
        
        # Load data
        embeddings = np.load(self.output_dir / f"{self.config.data.embedding_type}_embeddings.npy")
        labels = np.load(self.output_dir / f"{self.config.data.embedding_type}_labels.npy")
        
        # Generate visualizations
        self.visualizer.visualize_embeddings(
            embeddings, 
            labels,
            save_dir=self.output_dir
        ) 