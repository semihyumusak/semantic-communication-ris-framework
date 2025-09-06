# tasks/wireless_task.py

import torch

def compute_bit_error_rate(original_bits, recovered_bits):
    """
    Compute Bit Error Rate (BER) between original and recovered bits.

    Args:
        original_bits (torch.Tensor): Ground truth bits.
        recovered_bits (torch.Tensor): Demodulated bits.

    Returns:
        float: Bit error rate (0.0 to 1.0).
    """
    total_bits = original_bits.numel()
    bit_errors = (original_bits != recovered_bits).sum().item()
    ber = bit_errors / total_bits
    return ber

def compute_semantic_accuracy(original_bits, recovered_bits):
    """
    Compute semantic accuracy: 1 - BER

    Args:
        original_bits (torch.Tensor): Ground truth bits.
        recovered_bits (torch.Tensor): Demodulated bits.

    Returns:
        float: Semantic task accuracy.
    """
    ber = compute_bit_error_rate(original_bits, recovered_bits)
    accuracy = 1.0 - ber
    return accuracy
