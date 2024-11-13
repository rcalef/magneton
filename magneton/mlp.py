import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import pandas as pd

class SimpleMLPModel:
    def __init__(self, model_config: dict, device: torch.device):
        self.device = device
        self.num_layers = model_config.get("num_layers", 2)
        self.hidden_dim = model_config.get("hidden_dim", 256)
        self.dropout_rate = model_config.get("dropout_rate", 0.25)
        self.learning_rate = model_config.get("learning_rate", 5e-4)
        self.batch_size = model_config.get("batch_size", 64)
        self.num_steps = model_config.get("num_steps", 2000)
        self.validation_steps = model_config.get("validation_steps", 50)
        self.embeds = None  # Placeholder for embeddings
        self.model = None
        self.num_classes = model_config.get("num_classes", 100)

    def _init_model(self, input_dim: int, num_classes: int):
        """Initialize the model with the correct output dimension"""
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        # Output layer - num_classes outputs for multi-class classification
        layers.append(nn.Linear(self.hidden_dim, num_classes))
        
        self.model = nn.Sequential(*layers).to(self.device)

    def train_model(self, train_data, train_labels, val_data=None, val_labels=None):
        # Initialize model
        input_dim = train_data.shape[1]
        num_classes = self.num_classes
        self._init_model(input_dim, num_classes)

        # Convert to tensors and create DataLoader
        train_dataset = TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)  # Changed to long for classification
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        best_val_f1 = None
        best_model_state = None

        for step in range(self.num_steps):
            # Training step
            self.model.train()
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_data)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            # Validation step
            if val_data is not None and step % self.validation_steps == 0:
                metrics = self.evaluate_model(val_data, val_labels)
                print(f"Step {step}, Train Loss: {total_loss:.4f}, "
                      f"Val Acc: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")
                
                if best_val_f1 is None or metrics['f1'] > best_val_f1:
                    best_val_f1 = metrics['f1']
                    best_model_state = self.model.state_dict()

        # Restore the best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Best validation F1: {best_val_f1:.4f}")

    @torch.no_grad()
    def evaluate_model(self, data, labels):
        self.model.eval()
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        logits = self.model(data)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted')
        }

    def predict(self, data):
        self.model.eval()
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(data)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()