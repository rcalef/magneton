from pathlib import Path
from typing import Dict

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler

from magneton.config import TrainingConfig

class InterruptCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_exception(self, trainer, module, exception):
        raise exception

class ModelTrainer:
    """Trainer class that integrates with the pipeline"""

    def __init__(
        self,
        config: TrainingConfig,
        save_dir: str,
    ):
        self.config = config
        self.save_dir = save_dir
        self.model = None
        self.trainer = None

    def setup(
        self,
        model: L.LightningModule,
    ) -> None:
        """Set up model and trainer"""
        self.model = model

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
                patience=3
            ),
        ]

        # Set up logger
        if self.config.dev_run:
            logger = CSVLogger(
                save_dir=self.save_dir,
                name="csv_logger",
            )
            profiler = AdvancedProfiler(
                dirpath=self.save_dir,
                filename="profiler_output",
            )
            dev_run = 10
        else:
            logger = WandbLogger(
                entity="magneton",
                project="magneton",
                name=f"{model.name()}-training",
            )
            profiler = None
            dev_run = False

        # Create trainer
        self.trainer = L.Trainer(
            strategy="ddp",
            callbacks=callbacks,
            logger=logger,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            default_root_dir=self.save_dir,
            max_epochs=self.config.max_epochs,
            profiler=profiler,
            fast_dev_run=dev_run,
            **self.config.additional_training_kwargs,
        )

    def train_and_evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Train model and return metrics"""
        # Train model
        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Test model
        metrics = self.trainer.validate(self.model, dataloaders=val_loader)

        return metrics[0] if metrics else {}

    def save_model(self, save_path: Path):
        """Save model checkpoint"""
        if self.model is not None:
            self.trainer.save_checkpoint(save_path)

    def load_model(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        if self.model is not None:
            self.model = self.model.load_from_checkpoint(checkpoint_path)