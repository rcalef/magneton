import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the saved embeddings and labels
save_dir = "saved_embeddings"
embeddings = np.load(f"{save_dir}/protein_embeddings.npy")
labels = np.load(f"{save_dir}/protein_labels.npy")
element_names = np.load(f"{save_dir}/element_names.npy")

# Standardize the embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Create UMAP embedding
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings_scaled)

# Create a scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=labels,
    cmap='tab20',
    alpha=0.6,
    s=10
)

# Add a colorbar legend
plt.colorbar(scatter, label='Domain Types')
plt.title('UMAP visualization of protein domain embeddings')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Save the plot
plt.savefig(f"{save_dir}/umap_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

# Create a version with a sample of labeled points
plt.figure(figsize=(15, 10))

# Plot all points in grey first
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='lightgrey', alpha=0.5, s=5)

# Sample some points to label
n_samples = min(20, len(np.unique(labels)))  # Label up to 20 different domains
for i in np.random.choice(np.unique(labels), n_samples, replace=False):
    mask = labels == i
    points = embedding_2d[mask]
    centroid = points.mean(axis=0)
    plt.scatter(points[:, 0], points[:, 1], label=element_names[i], alpha=0.6, s=20)
    plt.annotate(element_names[i], centroid, alpha=0.7, fontsize=8)

plt.title('UMAP visualization of protein domain embeddings (with labels)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(f"{save_dir}/umap_visualization_labeled.png", dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics
print(f"Number of embeddings: {len(embeddings)}")
print(f"Number of unique domains: {len(np.unique(labels))}")
print(f"Embedding dimension: {embeddings.shape[1]}") 