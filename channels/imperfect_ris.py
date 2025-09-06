# channels/imperfect_ris.py

import torch
import numpy as np

class ImperfectRIS:
    def __init__(self, num_elements=256, ideal_gain_db=20, phase_error_std_deg=10):
        self.num_elements = num_elements
        self.ideal_gain_db = ideal_gain_db
        self.phase_error_std_deg = phase_error_std_deg

    def apply_ris_gain(self, snr_db):
        effective_snr_db = snr_db + self.ideal_gain_db
        return effective_snr_db

    def apply_phase_errors(self, signal):
        batch_size = signal.shape[0]
        phase_errors_deg = np.random.normal(0, self.phase_error_std_deg, size=batch_size)
        phase_errors_rad = np.deg2rad(phase_errors_deg)
        phase_rotation = torch.exp(1j * torch.tensor(phase_errors_rad, device=signal.device))
        return signal * phase_rotation
