# channels/awgn.py

import torch

def add_awgn_noise(signal, snr_db):
    """
    Add AWGN noise to the input signal based on a given SNR (in dB).

    Args:
        signal (torch.Tensor): Input semantic feature vector.
        snr_db (float): Signal-to-Noise Ratio in dB.

    Returns:
        torch.Tensor: Noisy signal.
    """
    # Calculate signal power and noise power
    signal_power = signal.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = torch.randn_like(signal) * noise_power.sqrt()

    # Add noise to the signal
    noisy_signal = signal + noise
    return noisy_signal
