# channels/rayleigh.py

import torch
import numpy as np


def add_rayleigh_fading(signal, snr_db):
    noise_std = 10 ** (-snr_db / 20)

    # Correct fading generation for 1D signals
    fading = (torch.randn_like(signal) + 1j * torch.randn_like(signal)) / np.sqrt(2)
    faded_signal = signal * fading

    # Add AWGN noise
    noise = noise_std * (torch.randn_like(signal) + 1j * torch.randn_like(signal)) / np.sqrt(2)
    return faded_signal + noise
