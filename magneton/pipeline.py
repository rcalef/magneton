import os

from pathlib import Path
from pprint import pprint

import torch

from magneton.config import PipelineConfig
from magneton.data import MagnetonDataModule
from magneton.data.core import get_substructure_parser
from magneton.evals.substructure_classification import classify_substructs
from magneton.evals.supervised_classification import run_supervised_classification
from magneton.training.trainer import ModelTrainer
from magneton.training.embedding_mlp import EmbeddingMLP, MultitaskEmbeddingMLP

class EmbeddingPipeline:
    """Main pipeline for protein embedding and analysis"""

    def __init__(self, cfg: PipelineConfig):
        print("\n=== Pipeline Configuration ===")
        print(f"Output Directory: {cfg.output_dir}")
        print(f"Test Directory: {cfg.test_dir}")
        print(f"Model Type: {cfg.model.model_type}")
        print(f"Model Checkpoint: {cfg.model.checkpoint}")
        print(f"Interpro Type: {cfg.data.substruct_types}")
        print(f"Full Config:")
        pprint(cfg, compact=False)
        print("============================\n")

        self.config = cfg
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_dir = Path(self.config.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt = self.config.model.checkpoint

    def run(self):
        """Run complete pipeline"""
        #self.run_embedding()
        self.run_training()
        self.run_evals()

    def run_embedding(self):
        """Generate and save embeddings"""
        raise ValueError("not implemented")

    def run_training(self):
        """Train and evaluate model using Lightning"""
        print("Training model...")
        assert self.config.training is not None, "No training config specified"
        # Initialize dataset
        data_module = MagnetonDataModule(
            data_config=self.config.data,
            model_type=self.config.embedding.model,
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training Device: {device}")

        # Not really a great way to get number of substructures being used without
        # reaching deep into the dataset which could be nested at different levels
        # depending on the model/config. So easiest to just get a new parser (which
        # really just counts the labels).
        substruct_parser = get_substructure_parser(self.config.data)
        # Train model
        if self.config.data.collapse_labels:
            model = EmbeddingMLP(
                config=self.config,
                num_classes=substruct_parser.num_labels(),
            )
        else:
            model = MultitaskEmbeddingMLP(
                config=self.config,
                num_classes=substruct_parser.num_labels(),
            )
            self.config.training.strategy = "ddp_find_unused_parameters_true"

        trainer = ModelTrainer(
            self.config.training,
            self.output_dir,
            self.config.run_id,
        )
        trainer.setup(model)

        metrics = trainer.train_and_evaluate(train_loader=train_loader, val_loader=val_loader)

        # Save model
        model_path = self.output_dir / f"model_{self.config.run_id}.pt"
        trainer.save_model(model_path)
        print(f"Saved model to {model_path}")
        print(f"Training metrics: {metrics}")

        return metrics

    def run_evals(self):
        """Generate visualizations and evals"""
        assert self.config.evaluate is not None, "No evaluation config specified"
        print("Evaluating and generating visualizations...")

        # Load model and data
        model = EmbeddingMLP.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
        )
        model.eval()

        for task in self.config.evaluate.tasks:
            print(f"{task} - evaluation start")
            if task == "substructure":
                data_module = MagnetonDataModule(
                    data_config=self.config.data,
                    model_type=self.config.embedding.model,
                )
                classify_substructs(
                    model,
                    data_module.test_dataloader(),
                )
            else:
                output_dir = os.path.join(self.config.output_dir, task)
                os.makedirs(output_dir, exist_ok=True)

                run_id = f"{self.config.run_id}_{task}"
                run_supervised_classification(
                    model=model,
                    task=task,
                    output_dir=output_dir,
                    run_id=run_id,
                    eval_config=self.config.evaluate,
                    data_config=self.config.data,
                )
