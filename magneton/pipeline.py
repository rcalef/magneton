from pathlib import Path
from pprint import pprint

import torch

from magneton.config import PipelineConfig
from magneton.data import MagnetonDataModule
from magneton.data.core import get_substructure_parser
from magneton.evaluations.substructure_classification import classify_substructs
from magneton.evaluations.supervised_classification import run_supervised_classification
from magneton.evaluations.zero_shot_evaluation import run_zero_shot_evaluation
from magneton.models.substructure_classifier import SubstructureClassifier
from magneton.training.trainer import ModelTrainer


class Pipeline:
    """Main pipeline for protein embedding and analysis"""

    def __init__(self, cfg: PipelineConfig):
        print("\n=== Pipeline Configuration ===")
        print(f"Output Directory: {cfg.output_dir}")
        print(f"Substructure types: {cfg.data.substruct_types}")
        print("Full Config:")
        pprint(cfg, compact=False)
        print("============================\n")

        self.config = cfg
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_training(self):
        """Train and evaluate model using Lightning"""
        print("Training model...")
        assert self.config.training is not None, "No training config specified"

        # Setup model
        substruct_parser = get_substructure_parser(self.config.data)
        model = SubstructureClassifier(
            config=self.config,
            num_classes=substruct_parser.num_labels(),
        )

        # Setup dataset
        want_distributed_sampler = torch.cuda.device_count() > 1
        data_module = MagnetonDataModule(
            data_config=self.config.data,
            model_type=self.config.base_model.model,
            distributed=want_distributed_sampler,
        )

        # Setup trainer
        if len(self.config.data.substruct_types) != 1:
            # With more than one substructure type, a given batch may only contain examples
            # of some substructure types, resulting in unused parameters.
            self.config.training.strategy = "ddp_find_unused_parameters_true"
        trainer = ModelTrainer(
            self.config.training,
            self.output_dir,
            self.config.run_id,
        )
        trainer.setup(model)

        # Train model
        metrics = trainer.train_and_evaluate(module=data_module)

        # Save model
        model_path = self.output_dir / f"model_{self.config.run_id}.pt"
        trainer.save_model(model_path)
        print(f"Saved model to {model_path}")
        print(f"Training metrics: {metrics}")

        return metrics

    def run_evals(self):
        """Generate visualizations and evals"""
        assert self.config.evaluate is not None, "No evaluation config specified"
        print("Evaluating...")

        for task in self.config.evaluate.tasks:
            print(f"{task} - evaluation start")
            run_id = f"{self.config.run_id}_{task}"
            output_dir = Path(self.config.output_dir) / task
            output_dir.mkdir(exist_ok=True)

            if task == "substructure":
                classify_substructs(
                    config=self.config,
                    output_dir=output_dir,
                )
            elif task == "zero_shot":
                run_zero_shot_evaluation(
                    config=self.config,
                    task=task,
                    output_dir=output_dir,
                    run_id=run_id,
                )
            else:
                run_supervised_classification(
                    config=self.config,
                    task=task,
                    output_dir=output_dir,
                    run_id=run_id,
                )
