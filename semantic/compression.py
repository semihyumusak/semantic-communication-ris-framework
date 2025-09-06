# semantic/compression.py

import torch
import torch.nn.functional as F

def compress_features(features, compression_ratio):
    """
    Compress semantic features by randomly zeroing out components to simulate compression.

    Args:
        features (torch.Tensor): Input semantic feature tensor (batch_size, feature_dim).
        compression_ratio (float): Fraction of features to keep (e.g., 0.25 means keep 25%).

    Returns:
        torch.Tensor: Compressed semantic feature tensor.
    """
    batch_size, feature_dim = features.shape
    keep_dim = int(feature_dim * compression_ratio)

    # Randomly select indices to keep
    indices = torch.randperm(feature_dim)[:keep_dim]
    mask = torch.zeros_like(features)
    mask[:, indices] = 1.0

    compressed_features = features * mask
    return compressed_features

def decompress_features(compressed_features):
    """
    Placeholder for decompression. In simple masking, decompression does nothing.
    For more advanced methods (e.g., autoencoders), decompression could be non-trivial.

    Args:
        compressed_features (torch.Tensor): Compressed feature tensor.

    Returns:
        torch.Tensor: Decompressed feature tensor (same as input here).
    """
    return compressed_features
