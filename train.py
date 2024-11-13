import numpy as np
import torch
from magneton.mlp import SimpleMLPModel
from sklearn.model_selection import train_test_split
import os

# Load the saved embeddings and labels
save_dir = "saved_embeddings"
embeddings = np.load(os.path.join(save_dir, "protein_embeddings.npy"))
labels = np.load(os.path.join(save_dir, "protein_labels.npy"))
element_names = np.load(os.path.join(save_dir, "element_names.npy"))

print(f"Loaded {len(element_names)} unique classes")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}")

# Add this after loading the labels but before the train_test_split
# Ensure labels are 0-indexed consecutive integers
unique_labels = np.unique(labels)

# Add this after remapping the labels
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")

label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_map[label] for label in labels])

print(f"Labels shape: {labels.shape}")
print(f"Label range: {labels.min()} to {labels.max()}")

X_train, X_val, y_train, y_val = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)

# Initialize model
model_config = {
    "num_layers": 2,
    "hidden_dim": 256,
    "dropout_rate": 0.25,
    "learning_rate": 5e-4,
    "batch_size": 64,
    "num_steps": 2000,
    "validation_steps": 50,
    "num_classes": num_classes
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = SimpleMLPModel(model_config, device)

# Train the model
mlp_model.train_model(X_train, y_train, X_val, y_val)

# Final evaluation
metrics = mlp_model.evaluate_model(X_val, y_val)
print(f"Final validation metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

# Optional: Save the predictions
# val_preds = mlp_model.predict(X_val)
# np.save(os.path.join(save_dir, "validation_predictions.npy"), val_preds) 