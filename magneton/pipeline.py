from pathlib import Path
from pprint import pprint

import torch
import torch.distributed as dist
import torch.nn.parallel
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
        print(f"Test Directory: {cfg.test_dir}")
        print(f"Model Type: {cfg.model.model_type}")
        print(f"Model Checkpoint: {cfg.model.checkpoint}")
        print(f"Full Config:")
        pprint(cfg, compact=False)
        print("============================\n")

        self.config = cfg
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_dir = Path(self.config.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt = self.config.model.checkpoint

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
        pass

        # TODO
        # Check if embeddings exist, terminate if so
        # Implement override if want to regenerate

        # embedder  = EmbedderFactory.create_embedder(self.config.embedding)

        # # Get data loader
        # loader = self.data_module.predict_dataloader()

        # # Get embeddings and associated IDs
        # all_embeds = []
        # for batch in tqdm(loader, desc=f"{embedder.model_name()} embedding"):
        #     batch = batch.to(self.config.embedding.device)
        #     batch_embeds = embedder.embed_batch(batch)
        #     all_embeds.append(batch_embeds.detach().cpu())

        # # Save results
        # embedding_path = self.output_dir / f"{embedder.model_name()}.embeddings.pt"
        # all_embeds = torch.cat(all_embeds, dim=0).cpu()
        # torch.save(all_embeds, embedding_path)

        # print("Done generating embeddings")

    def run_training(self):
        """Train and evaluate model using Lightning"""
        print("Training model...")
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training Device: {device}")

        # Train model
        model = EmbeddingMLP(
            config=self.config,
            num_classes=len(train_loader.dataset.substruct_parser.type_to_label),
            device=device
        )
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model.to(device)

        trainer = ModelTrainer(self.config.training, self.output_dir)
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
        print("Evaluating and generating visualizations...")

        # Load model and data
        model = EmbeddingMLP.load_from_checkpoint(
            self.ckpt,
        )
        model.eval()

        # UMAP
        # Get embeddings for all substructures
        all_embeddings = []

        results = {
            'protein_ids': [],
            'true_labels': [],
            'true_mappings': [],
            'pred_labels': [],  # Store argmax predictions
            'pred_mappings': [],
            'top_k_labels': [],  # Store top-k predictions
            'top_k_probs': []   # Store top-k probabilities
        }
        accs = []
        k = 3
        # TODO Change to selected_subset tsv once moved to MLP trained on the right number of classes
        # labels_tsv_path = '/weka/scratch/weka/kellislab/rcalef/data/interpro/103.0/label_sets/selected_subset/Conserved_site.labels.tsv'
        labels_tsv_path = '/weka/scratch/weka/kellislab/rcalef/data/interpro/103.0/label_sets/Conserved_site.labels.tsv'
        labels_df = pd.read_csv(labels_tsv_path, sep='\t')
        label_to_element = dict(zip(labels_df['label'], labels_df['element_name']))

        with torch.no_grad():
            for batch in tqdm(self.data_module.test_dataloader(), desc="Evaluating batches"):
                # Skip batch if no substructs
                # TODO Reevaluate how to deal with batches that have only proteins with no substructures
                if [] in batch.substructures:
                    continue

                # Get substructure embeddings
                substruct_embeds = model.embed(batch)
                all_embeddings.append(substruct_embeds)
                
                # Get protein id
                # batch_ids = batch.prot_ids
                batch_ids = [batch.prot_ids[i] for i, prot_substructs in enumerate(batch.substructures)
                        for substruct in prot_substructs]
                
                # Get labels
                batch_labels = [substruct.label for prot_substructs in batch.substructures 
                            for substruct in prot_substructs]

                # Forward pass
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)
                preds = torch.argmax(logits, dim=1)

                # Map from id to class
                true_mappings = [label_to_element.get(label, f"Unknown label: {label}") for label in batch_labels]
                pred_mappings = [label_to_element.get(label, f"Unknown label: {label}") for label in [int(x.item()) for x in preds]]

                # Logging metrics
                results['protein_ids'].extend(batch_ids)
                results['true_labels'].extend(batch_labels)
                results['true_mappings'].extend(true_mappings)

                # results['pred_probs'].extend([str(x) for x in probs])
                results['pred_labels'].extend([int(x.item()) for x in preds])
                results['pred_mappings'].extend(pred_mappings)

                results['top_k_labels'].extend(topk_indices.cpu().numpy().tolist())
                results['top_k_probs'].extend(topk_probs.cpu().numpy().tolist())

                accs.append(model.train_acc(preds, torch.tensor(batch_labels, device=logits.device)))
        
        # Combine embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Create UMAP visualization
        visualizer = UMAPVisualizer()
        visualizer.visualize_embeddings(
            all_embeddings,
            results['pred_labels'],
            save_dir=self.test_dir / "visualizations",
            title=f"UMAP of {self.config.embedding.model} Substructure Embeddings With Predicted Labels",
            type="pred",
        )
        visualizer.visualize_embeddings(
            all_embeddings,
            results['true_labels'],
            save_dir=self.test_dir / "visualizations",
            title=f"UMAP of {self.config.embedding.model} Substructure Embeddings",
            type="true",
        )
          
        # Write results
        df = pd.DataFrame(results)
        csv_path = self.test_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)

        print(f"Accuracies for each batch: {accs}")
        print(f"Saved detailed evaluation results to {self.test_dir}")