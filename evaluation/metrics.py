# evaluation/metrics.py

import torch
import torch.nn.functional as F

def compute_bit_rate(compression_ratio, original_dim):
    """
    Compute the effective bit rate based on compression.

    Args:
        compression_ratio (float): Fraction of features kept after compression.
        original_dim (int): Original semantic feature dimension.

    Returns:
        float: Effective bit rate (normalized).
    """
    return compression_ratio * original_dim

def compute_distortion(original_features, reconstructed_features):
    """
    Compute distortion between original and reconstructed semantic features.

    Args:
        original_features (torch.Tensor): Original semantic features.
        reconstructed_features (torch.Tensor): Reconstructed semantic features.

    Returns:
        float: Mean Squared Error (distortion).
    """
    mse = F.mse_loss(reconstructed_features, original_features)
    return mse.item()

def compute_relevance(predictions, targets):
    """
    Compute semantic relevance based on task success (e.g., classification accuracy).

    Args:
        predictions (torch.Tensor): Model outputs (logits).
        targets (torch.Tensor): Ground truth labels.

    Returns:
        float: Classification accuracy (0.0 to 1.0).
    """
    predicted_labels = torch.argmax(predictions, dim=1)
    correct = (predicted_labels == targets).float().sum()
    total = targets.shape[0]
    accuracy = correct / total
    return accuracy.item()
