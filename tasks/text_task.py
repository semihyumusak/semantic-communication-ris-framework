# tasks/text_task.py

import torch

def generate_text_classification_data(batch_size, input_dim, num_classes=4):
    """
    Simulate simple text classification data.

    Args:
        batch_size (int): Number of samples.
        input_dim (int): Input dimension (simulated as embedding size).
        num_classes (int): Number of classes.

    Returns:
        X (torch.Tensor): Simulated input embeddings.
        targets (torch.Tensor): Simulated labels.
    """
    X = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, num_classes, (batch_size,))
    return X, targets
