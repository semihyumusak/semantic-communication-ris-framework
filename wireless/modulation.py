# wireless/modulation.py

import torch

def qpsk_modulate(bits):
    """
    QPSK modulation: Map 2 bits to 1 complex symbol.
    Args:
        bits (torch.Tensor): 1D tensor of 0s and 1s, length divisible by 2.
    Returns:
        torch.Tensor: Complex symbols (real + j*imag).
    """
    assert bits.numel() % 2 == 0, "Number of bits must be even for QPSK."
    bits = bits.view(-1, 2)
    real = 2 * bits[:, 0] - 1
    imag = 2 * bits[:, 1] - 1
    symbols = real + 1j * imag
    symbols = symbols / (2**0.5)  # Normalize average power to 1
    return symbols

def qam16_modulate(bits):
    """
    16-QAM modulation: Map 4 bits to 1 complex symbol.
    Args:
        bits (torch.Tensor): 1D tensor of 0s and 1s, length divisible by 4.
    Returns:
        torch.Tensor: Complex symbols (real + j*imag).
    """
    assert bits.numel() % 4 == 0, "Number of bits must be multiple of 4 for 16-QAM."
    bits = bits.view(-1, 4)
    real = (2 * bits[:, 0] + bits[:, 1]) * 2 - 3
    imag = (2 * bits[:, 2] + bits[:, 3]) * 2 - 3
    symbols = real + 1j * imag
    symbols = symbols / (10**0.5)  # Normalize average power to 1
    return symbols

def modulate(bits, modulation_scheme='qpsk'):
    """
    General modulate function.
    Args:
        bits (torch.Tensor): Input bits.
        modulation_scheme (str): 'qpsk' or 'qam16'
    Returns:
        torch.Tensor: Complex symbols.
    """
    if modulation_scheme == 'qpsk':
        return qpsk_modulate(bits)
    elif modulation_scheme == 'qam16':
        return qam16_modulate(bits)
    else:
        raise ValueError("Unsupported modulation scheme. Choose 'qpsk' or 'qam16'.")
