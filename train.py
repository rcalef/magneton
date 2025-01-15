import numpy as np
import torch
from magneton.mlp import SimpleMLPModel
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

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

# Add after loading the data but before training
# Analyze class distribution
unique, counts = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))

print("\nClass Distribution:")
for label, count in class_distribution.items():
    class_name = element_names[label]
    percentage = (count / len(labels)) * 100
    print(f"Class {label} ({class_name}): {count} samples ({percentage:.2f}%)")

# Optional: Plot class distribution
plt.figure(figsize=(15, 5))
plt.bar(range(len(counts)), counts)
plt.title('Class Distribution')
plt.xlabel('Class Index')
plt.ylabel('Number of Samples')
plt.savefig(os.path.join(save_dir, "class_distribution.png"))
plt.close()

# Train the model
mlp_model.train_model(X_train, y_train, X_val, y_val)

# Final evaluation
metrics = mlp_model.evaluate_model(X_val, y_val)
print(f"Final validation metrics:") 
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

def plot_class_distributions(true_dist, pred_dist, element_names, save_dir):
    """Create side-by-side bar plots comparing true and predicted class distributions."""
    plt.figure(figsize=(20, 10))
    
    # Number of classes
    n_classes = len(true_dist)
    
    # Set up the bar positions
    bar_width = 0.35
    r1 = np.arange(n_classes)
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, true_dist, width=bar_width, label='True Distribution', color='skyblue')
    plt.bar(r2, pred_dist, width=bar_width, label='Predicted Distribution', color='lightcoral')
    
    # Customize the plot
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('True vs Predicted Class Distribution')
    
    # Show only every 20th tick
    tick_positions = [r + bar_width/2 for r in range(0, n_classes, 20)]
    tick_labels = [str(i) for i in range(0, n_classes, 20)]
    plt.xticks(tick_positions, tick_labels)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distribution_comparison.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_predictions(y_true, y_pred, element_names, save_dir):
    """Analyze and visualize prediction distribution and accuracy per class"""
    # Get distributions
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    true_unique, true_counts = np.unique(y_true, return_counts=True)
    
    # Create full distribution arrays (including zeros for missing classes)
    n_classes = len(element_names)
    true_dist = np.zeros(n_classes)
    pred_dist = np.zeros(n_classes)
    
    true_dist[true_unique] = true_counts
    pred_dist[pred_unique] = pred_counts
    
    # Plot distributions
    plot_class_distributions(true_dist, pred_dist, element_names, save_dir)
    
    # Calculate per-class accuracy
    accuracies = []
    for label in range(n_classes):
        mask = (y_true == label)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_true[mask])
            accuracies.append(class_acc)
        else:
            accuracies.append(0)
    
    # Plot per-class accuracy
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(accuracies)), accuracies)
    plt.title('Per-class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Show only every 20th tick
    tick_positions = range(0, len(accuracies), 20)
    tick_labels = [str(i) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print numerical results with both class number and name
    print("\nPrediction Distribution:")
    for label, count in zip(pred_unique, pred_counts):
        percentage = (count / len(y_pred)) * 100
        print(f"Class {label} ({element_names[label]}): {count} predictions ({percentage:.2f}%)")
    
    print("\nPer-class Accuracy:")
    for label in range(n_classes):
        if true_dist[label] > 0:
            print(f"Class {label} ({element_names[label]}): {accuracies[label]:.4f}")

# After training, generate the analysis
val_preds = mlp_model.predict(X_val)
val_pred_classes = np.argmax(val_preds, axis=1)
analyze_predictions(y_val, val_pred_classes, element_names, save_dir)

# Optional: Save the predictions
np.save(os.path.join(save_dir, "validation_predictions.npy"), val_preds) 