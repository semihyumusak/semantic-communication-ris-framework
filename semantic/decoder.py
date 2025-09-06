# semantic/decoder.py

import torch
import torch.nn as nn

class SemanticDecoder(nn.Module):
    """
    Simple MLP-based semantic decoder that maps semantic features Z_hat to task output (e.g., class labels).
    """

    def __init__(self, feature_dim=64, output_dim=10):
        super(SemanticDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z_hat):
        return self.decoder(z_hat)

def create_decoder(feature_dim=64, output_dim=10):
    """
    Utility function to create a SemanticDecoder with given dimensions.

    Args:
        feature_dim (int): Dimension of semantic feature vector.
        output_dim (int): Number of output classes.

    Returns:
        SemanticDecoder: Initialized decoder model.
    """
    return SemanticDecoder(feature_dim, output_dim)
