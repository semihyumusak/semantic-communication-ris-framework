# tasks/image_task.py

import torch

def generate_image_classification_data(batch_size, input_dim, num_classes=10):
    """
    Simulate simple image classification data.

    Args:
        batch_size (int): Number of samples.
        input_dim (int): Input dimension (e.g., flattened 28x28 image).

    Returns:
        X (torch.Tensor): Simulated image data.
        targets (torch.Tensor): Simulated labels.
    """
    X = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, num_classes, (batch_size,))
    return X, targets
