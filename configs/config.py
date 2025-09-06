# configs/config.py
import numpy as np
class Config:
    def __init__(self):
        self.random_seed = 42
        self.device = "cpu"
        self.snr_db_range = np.arange(-30, 30, 0.1) # Stronger SNR range

config = Config()
