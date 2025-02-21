from pathlib import Path
from typing import Dict

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

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
            # ModelCheckpoint(
            #     monitor="val_f1",
            #     mode="max",
            #     save_top_k=3,
            #     filename="{epoch}-{val_f1:.2f}"
            # ),
            # EarlyStopping(
            #     monitor="val_f1",
            #     mode="max",
            #     patience=10
            # ),
            InterruptCallback(),
        ]

        # Set up logger
        logger = WandbLogger(
            project="magneton",
            name=f"{model.name()}-training"
        )
        # logger = CSVLogger(
        #     save_dir=self.save_dir,
        # )

        # Create trainer
        self.trainer = L.Trainer(
            callbacks=callbacks,
            logger=logger,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            default_root_dir=self.save_dir,
            max_epochs=self.config.max_epochs,
#            fast_dev_run=10,
            **self.config.additional_training_kwargs,
            # max_epochs=self.config.training.max_epochs,
            # precision=self.config.training.precision,
            # accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            # gradient_clip_val=self.config.training.gradient_clip_val
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