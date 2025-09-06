# channels/rician.py

import torch

def add_rician_fading(signal, snr_db, K_factor=10):
    """
    Add Rician fading and AWGN noise to the signal.

    Args:
        signal (torch.Tensor): Input semantic feature vector.
        snr_db (float): Signal-to-Noise Ratio in dB.
        K_factor (float): Rician K-factor (ratio of LOS power to scattered power).

    Returns:
        torch.Tensor: Faded and noisy signal.
    """
    batch_size, feature_dim = signal.shape
    s = (K_factor / (K_factor + 1)) ** 0.5
    sigma = (1 / (2 * (K_factor + 1))) ** 0.5

    h = s + sigma * torch.randn(batch_size, feature_dim)  # Rician fading coefficients

    faded_signal = signal * h

    # AWGN noise
    signal_power = faded_signal.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(faded_signal) * noise_power.sqrt()

    return faded_signal + noise
