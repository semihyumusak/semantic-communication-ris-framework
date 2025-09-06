# channels/mixed_channel.py

import torch
from channels.awgn import add_awgn_noise
from channels.rayleigh import add_rayleigh_fading

def apply_mixed_channel(signal, snr_db, fading_factor=0.7):
    faded_signal = add_rayleigh_fading(signal, snr_db)
    mixed_signal = (1 - fading_factor) * signal + fading_factor * faded_signal
    mixed_signal = add_awgn_noise(mixed_signal, snr_db)
    return mixed_signal
