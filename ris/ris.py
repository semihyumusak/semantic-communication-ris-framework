# ris/ris.py

class RIS:
    def __init__(self, num_elements=256, ideal_gain_db=10):  # Stronger RIS gain
        self.num_elements = num_elements
        self.ideal_gain_db = ideal_gain_db

    def apply_ris_gain(self, snr_db):
        effective_snr_db = snr_db + self.ideal_gain_db
        return effective_snr_db
