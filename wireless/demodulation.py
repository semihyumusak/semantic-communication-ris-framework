# wireless/demodulation.py

import torch

def qpsk_demodulate(symbols):
    """
    QPSK demodulation: Map 1 complex symbol to 2 bits.
    Args:
        symbols (torch.Tensor): Complex symbols.
    Returns:
        torch.Tensor: 1D tensor of bits.
    """
    real = symbols.real
    imag = symbols.imag
    bits_real = (real >= 0).long()
    bits_imag = (imag >= 0).long()
    bits = torch.stack([bits_real, bits_imag], dim=1).view(-1)
    return bits

def qam16_demodulate(symbols):
    """
    16-QAM demodulation: Map 1 complex symbol to 4 bits.
    Args:
        symbols (torch.Tensor): Complex symbols.
    Returns:
        torch.Tensor: 1D tensor of bits.
    """
    real = symbols.real * (10**0.5)
    imag = symbols.imag * (10**0.5)

    bits_real_msb = (real > 0).long()
    bits_real_lsb = ((real.abs() < 2)).long()

    bits_imag_msb = (imag > 0).long()
    bits_imag_lsb = ((imag.abs() < 2)).long()

    bits = torch.stack([bits_real_msb, bits_real_lsb, bits_imag_msb, bits_imag_lsb], dim=1).view(-1)
    return bits

def demodulate(symbols, modulation_scheme='qpsk'):
    """
    General demodulate function.
    Args:
        symbols (torch.Tensor): Complex symbols.
        modulation_scheme (str): 'qpsk' or 'qam16'
    Returns:
        torch.Tensor: Bits.
    """
    if modulation_scheme == 'qpsk':
        return qpsk_demodulate(symbols)
    elif modulation_scheme == 'qam16':
        return qam16_demodulate(symbols)
    else:
        raise ValueError("Unsupported modulation scheme. Choose 'qpsk' or 'qam16'.")
