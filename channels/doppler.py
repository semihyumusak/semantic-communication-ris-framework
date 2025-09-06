# channels/doppler.py

import torch
import numpy as np

def apply_doppler_shift(signal, max_doppler_hz=50, symbol_duration_s=1e-3):
    batch_size = signal.shape[0]
    doppler_shifts_hz = np.random.uniform(-max_doppler_hz, max_doppler_hz, size=batch_size)
    doppler_phases = 2 * np.pi * doppler_shifts_hz * symbol_duration_s
    doppler_rotation = torch.exp(1j * torch.tensor(doppler_phases, device=signal.device))
    return signal * doppler_rotation
