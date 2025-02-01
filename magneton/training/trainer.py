from typing import Dict, Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path

from .lightning_model import ProteinClassifier
from .data_module import ProteinDataModule
from ..config.config import Config

class ModelTrainer:
    """Trainer class that integrates with the pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.trainer = None
        
    def setup(self, num_classes: int, input_dim: int):
        """Set up model and trainer"""
        # Create model
        self.model = ProteinClassifier(
            config=self.config.model,
            num_classes=num_classes,
            input_dim=input_dim
        )
        
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                monitor="val_f1",
                mode="max",
                save_top_k=3,
                filename="{epoch}-{val_f1:.2f}"
            ),
            EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=10
            )
        ]
        
        # Set up logger
        logger = WandbLogger(
            project="protein-embeddings",
            name=f"{self.config.data.embedding_type}-training"
        )
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            accelerator=self.config.training.accelerator,
            devices=self.config.training.devices,
            precision=self.config.training.precision,
            callbacks=callbacks,
            logger=logger,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            gradient_clip_val=self.config.training.gradient_clip_val
        )
        
    def train_and_evaluate(self, datamodule: ProteinDataModule) -> Dict[str, float]:
        """Train model and return metrics"""
        # Train model
        self.trainer.fit(self.model, datamodule)
        
        # Test model
        metrics = self.trainer.test(self.model, datamodule)
        
        return metrics[0] if metrics else {}
        
    def save_model(self, save_path: Path):
        """Save model checkpoint"""
        if self.model is not None:
            self.trainer.save_checkpoint(save_path)
            
    def load_model(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        if self.model is not None:
            self.model = self.model.load_from_checkpoint(checkpoint_path) 