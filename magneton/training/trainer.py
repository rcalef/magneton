from pathlib import Path
from typing import Dict

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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
        run_id: str,
    ):
        self.config = config
        self.save_dir = save_dir
        self.run_id = run_id
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
                dirpath=self.save_dir / f"checkpoints_{self.run_id}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                filename="{epoch}-{val_loss:.2f}",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ]

        # Set up logger
        is_dev_run = (type(self.config.dev_run) is int) or bool(self.config.dev_run)
        if is_dev_run:
            logger = CSVLogger(
                save_dir=self.save_dir,
                name="csv_logger",
            )
        else:
            logger = WandbLogger(
                entity="magneton",
                project="magneton",
                name=self.run_id,
            )

        if self.config.profile:
            profiler = AdvancedProfiler(
                dirpath=self.save_dir,
                filename="profiler_output",
            )
        else:
            profiler = None

        # Create trainer
        self.trainer = L.Trainer(
            strategy=self.config.strategy,
            callbacks=callbacks,
            logger=logger,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            default_root_dir=self.save_dir,
            max_epochs=self.config.max_epochs,
            profiler=profiler,
            fast_dev_run=self.config.dev_run,
            precision=self.config.precision,
            **self.config.additional_training_kwargs,
        )

    def train_and_evaluate(
        self,
        module: L.LightningDataModule,
    ) -> Dict[str, float]:
        """Train model and return metrics"""
        # Train model
        has_pretrained_fisher = self.config.reuse_ewc_weights is not None
        if self.config.loss_strategy == "ewc" and not has_pretrained_fisher:
            self.model.calc_fisher_state = True

            self.trainer.predict(
                self.model,
                datamodule=module,
                return_predictions=False,
            )

            self.model.calc_fisher_state = False

        self.trainer.fit(
            self.model,
            datamodule=module,
        )

        # Test model
        metrics = self.trainer.validate(self.model, datamodule=module)

        return metrics[0] if metrics else {}

    def save_model(self, save_path: Path):
        """Save model checkpoint"""
        if self.model is not None:
            self.trainer.save_checkpoint(save_path)

    def load_model(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        if self.model is not None:
            self.model = self.model.load_from_checkpoint(checkpoint_path)
