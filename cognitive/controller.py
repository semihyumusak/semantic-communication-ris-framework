# cognitive/controller.py

class CognitiveController:
    """
    Simple cognitive controller that adapts compression ratio based on semantic relevance.
    """

    def __init__(self, initial_compression_ratio=0.25, min_compression=0.1, max_compression=0.5, relevance_threshold=0.7):
        self.compression_ratio = initial_compression_ratio
        self.min_compression = min_compression
        self.max_compression = max_compression
        self.relevance_threshold = relevance_threshold

    def update(self, current_relevance):
        """
        Update compression ratio based on current relevance.

        Args:
            current_relevance (float): Current semantic relevance (accuracy).

        Returns:
            float: Updated compression ratio.
        """
        if current_relevance < self.relevance_threshold:
            # If relevance is low, decrease compression (i.e., send more info)
            self.compression_ratio = min(self.compression_ratio + 0.05, self.max_compression)
        else:
            # If relevance is high, increase compression (i.e., send fewer features)
            self.compression_ratio = max(self.compression_ratio - 0.05, self.min_compression)
        return self.compression_ratio
