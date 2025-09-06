# semantic/encoder.py

import torch
import torch.nn as nn

class SemanticEncoder(nn.Module):
    """
    Simple MLP-based semantic encoder that maps input signal X to semantic features Z.
    """

    def __init__(self, input_dim=784, feature_dim=64):
        super(SemanticEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def create_encoder(input_dim=784, feature_dim=64):
    """
    Utility function to create a SemanticEncoder with given dimensions.

    Args:
        input_dim (int): Input dimension (e.g., 784 for flattened 28x28 image).
        feature_dim (int): Dimension of semantic feature vector.

    Returns:
        SemanticEncoder: Initialized encoder model.
    """
    return SemanticEncoder(input_dim, feature_dim)
