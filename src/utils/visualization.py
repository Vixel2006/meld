import matplotlib.pyplot as plt
import os

def plot_losses_vs_epochs(losses: dict, save_path: str = None):
    plt.figure(figsize=(10, 6))
    for loss_name, loss_values in losses.items():
        plt.plot(loss_values, label=loss_name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss vs. Epochs plot saved to {save_path}")
    else:
        plt.show()

def plot_loss_vs_concept_dimensions(concept_dims_losses: list, save_path: str = None):
    if not concept_dims_losses:
        print("No data to plot for Loss vs. Concept Dimensions.")
        return

    concept_dims = [item[0] for item in concept_dims_losses]
    losses = [item[1] for item in concept_dims_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(concept_dims, losses, marker='o', linestyle='-')

    plt.xlabel("Concept Vector Dimension")
    plt.ylabel("Loss")
    plt.title("Loss vs. Concept Vector Dimensions")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss vs. Concept Dimensions plot saved to {save_path}")
    else:
        plt.show()

def plot_concept_sparsity(sparsity_data: dict, save_path: str = None):
    if not sparsity_data:
        print("No data to plot for Concept Sparsity.")
        return

    concept_ids = list(sparsity_data.keys())
    sparsity_scores = list(sparsity_data.values())

    plt.figure(figsize=(12, 6))
    plt.bar(concept_ids, sparsity_scores, color='skyblue')
    plt.xlabel("Concept ID")
    plt.ylabel("Sparsity Score")
    plt.title("Concept Sparsity Distribution")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Concept Sparsity plot saved to {save_path}")
    else:
        plt.show()

def plot_cross_modal_similarity(similarity_matrix, labels: tuple, save_path: str = None):
    if similarity_matrix is None or not labels:
        print("No data or labels to plot for Cross-Modal Similarity.")
        return

    row_labels, col_labels = labels
    
    plt.figure(figsize=(len(col_labels) * 0.8, len(row_labels) * 0.8)) # Adjust figure size dynamically
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Similarity Score")
    plt.xticks(range(len(col_labels)), col_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(row_labels)), row_labels, fontsize=8)
    plt.xlabel("Text Embeddings" if col_labels and "text" in col_labels[0].lower() else "Column Labels")
    plt.ylabel("Image Embeddings" if row_labels and "image" in row_labels[0].lower() else "Row Labels")
    plt.title("Cross-Modal Similarity Matrix")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Cross-Modal Similarity plot saved to {save_path}")
    else:
        plt.show()

def plot_retrieval_metrics(metrics_data: dict, save_path: str = None):
    if not metrics_data:
        print("No data to plot for Retrieval Metrics.")
        return

    metric_names = list(metrics_data.keys())
    metric_values = list(metrics_data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metric_values, color='lightcoral')
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Retrieval Metrics")
    plt.ylim(0, 1.0) # Assuming scores are between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Retrieval Metrics plot saved to {save_path}")
    else:
        plt.show()

def plot_generative_metrics(metrics_data: dict, save_path: str = None):
    if not metrics_data:
        print("No data to plot for Generative Metrics.")
        return

    metric_names = list(metrics_data.keys())
    metric_values = list(metrics_data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metric_values, color='lightgreen')
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Generative Metrics")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Generative Metrics plot saved to {save_path}")
    else:
        plt.show()
