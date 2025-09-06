# cognitive/adaptive_control.py

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable


@dataclass
class AdaptiveControlConfig:
    """Configuration parameters for adaptive control mechanisms"""
    initial_compression_ratio: float = 0.5
    min_compression_ratio: float = 0.1
    max_compression_ratio: float = 0.9
    relevance_threshold: float = 0.7
    adaptation_rate: float = 0.05
    window_size: int = 5
    metric_weights: Dict[str, float] = None
    channel_adaptation: bool = True
    snr_adaptation: bool = True
    energy_aware: bool = True


class EnhancedCognitiveController:
    """
    Enhanced cognitive controller that adapts compression ratio based on multiple
    semantic metrics, channel conditions, and energy constraints.
    """

    def __init__(self, config: AdaptiveControlConfig = None):
        """
        Initialize the enhanced cognitive controller.

        Args:
            config: Configuration parameters for the controller
        """
        self.config = config or AdaptiveControlConfig()
        self.compression_ratio = self.config.initial_compression_ratio
        self.history = {
            'compression_ratios': [],
            'relevance_scores': [],
            'snr_values': [],
            'energy_usage': [],
            'channel_conditions': []
        }

        # Default metric weights if not provided
        if self.config.metric_weights is None:
            self.config.metric_weights = {
                'bit_semantic_accuracy': 0.5,
                'cosine_similarity': 0.3,
                'mutual_information': 0.2
            }

        # Normalize weights
        total = sum(self.config.metric_weights.values())
        for k in self.config.metric_weights:
            self.config.metric_weights[k] /= total

    def update(self, semantic_metrics: Dict[str, float],
               snr_db: Optional[float] = None,
               channel_type: Optional[str] = None,
               energy_usage: Optional[float] = None) -> float:
        """
        Update compression ratio based on current performance.

        Args:
            semantic_metrics: Dictionary of semantic metrics
            snr_db: Current signal-to-noise ratio (dB)
            channel_type: Type of channel (e.g., 'awgn', 'doppler')
            energy_usage: Current energy usage

        Returns:
            Updated compression ratio
        """
        # Calculate weighted relevance score
        relevance_score = self._calculate_relevance_score(semantic_metrics)

        # Record history
        self.history['compression_ratios'].append(self.compression_ratio)
        self.history['relevance_scores'].append(relevance_score)
        if snr_db is not None:
            self.history['snr_values'].append(snr_db)
        if channel_type is not None:
            self.history['channel_conditions'].append(channel_type)
        if energy_usage is not None:
            self.history['energy_usage'].append(energy_usage)

        # Limit history length
        for key in self.history:
            if len(self.history[key]) > self.config.window_size:
                self.history[key] = self.history[key][-self.config.window_size:]

        # Basic relevance-based adaptation
        if relevance_score < self.config.relevance_threshold:
            # If relevance is low, decrease compression (i.e., send more info)
            self.compression_ratio = min(
                self.compression_ratio + self.config.adaptation_rate,
                self.config.max_compression_ratio
            )
        else:
            # If relevance is high, increase compression (i.e., send fewer features)
            self.compression_ratio = max(
                self.compression_ratio - self.config.adaptation_rate,
                self.config.min_compression_ratio
            )

        # Apply channel-based adaptation if enabled
        if self.config.channel_adaptation and channel_type:
            self.compression_ratio = self._adapt_to_channel(
                self.compression_ratio, channel_type
            )

        # Apply SNR-based adaptation if enabled
        if self.config.snr_adaptation and snr_db is not None:
            self.compression_ratio = self._adapt_to_snr(
                self.compression_ratio, snr_db
            )

        # Apply energy-aware adaptation if enabled
        if self.config.energy_aware and energy_usage is not None:
            self.compression_ratio = self._adapt_to_energy(
                self.compression_ratio, energy_usage
            )

        return self.compression_ratio

    def _calculate_relevance_score(self, semantic_metrics: Dict[str, float]) -> float:
        """
        Calculate weighted relevance score from multiple semantic metrics.

        Args:
            semantic_metrics: Dictionary of semantic metrics

        Returns:
            Weighted relevance score
        """
        score = 0.0
        weight_sum = 0.0

        for metric_name, metric_value in semantic_metrics.items():
            if metric_name in self.config.metric_weights:
                weight = self.config.metric_weights[metric_name]
                score += weight * metric_value
                weight_sum += weight

        # Normalize by applied weights
        if weight_sum > 0:
            score /= weight_sum

        return score

    def _adapt_to_channel(self, current_ratio: float, channel_type: str) -> float:
        """
        Adapt compression ratio based on channel type.

        Args:
            current_ratio: Current compression ratio
            channel_type: Type of channel

        Returns:
            Adjusted compression ratio
        """
        # Channel-specific adaptations
        channel_factors = {
            'awgn': 1.0,  # Standard case
            'doppler': 0.8,  # More conservative with Doppler
            'rayleigh': 0.7,  # More conservative with Rayleigh
            'rician': 0.85,  # Somewhat conservative with Rician
            'mixed': 0.6,  # Most conservative with mixed channel
            'imperfect_ris': 0.9  # Slightly conservative with imperfect RIS
        }

        # Apply channel factor (default to 1.0 if unknown)
        factor = channel_factors.get(channel_type.lower(), 1.0)

        # Adjust compression ratio
        new_ratio = current_ratio * factor

        # Ensure within bounds
        return max(min(new_ratio, self.config.max_compression_ratio),
                   self.config.min_compression_ratio)

    def _adapt_to_snr(self, current_ratio: float, snr_db: float) -> float:
        """
        Adapt compression ratio based on SNR.

        Args:
            current_ratio: Current compression ratio
            snr_db: Current SNR in dB

        Returns:
            Adjusted compression ratio
        """
        # SNR-based adaptation logic
        if snr_db < -10:
            # Very low SNR - be very conservative
            factor = 0.6
        elif snr_db < 0:
            # Low SNR - be somewhat conservative
            factor = 0.8
        elif snr_db < 10:
            # Moderate SNR - be slightly conservative
            factor = 0.9
        elif snr_db < 20:
            # Good SNR - be neutral
            factor = 1.0
        else:
            # Excellent SNR - be more aggressive
            factor = 1.1

        # Apply SNR factor
        new_ratio = current_ratio * factor

        # Ensure within bounds
        return max(min(new_ratio, self.config.max_compression_ratio),
                   self.config.min_compression_ratio)

    def _adapt_to_energy(self, current_ratio: float, energy_usage: float) -> float:
        """
        Adapt compression ratio based on energy usage.

        Args:
            current_ratio: Current compression ratio
            energy_usage: Current energy usage (normalized)

        Returns:
            Adjusted compression ratio
        """
        # Only adapt if we have enough history
        if len(self.history['energy_usage']) < 2:
            return current_ratio

        # Get energy usage trend
        recent_energy = self.history['energy_usage'][-self.config.window_size:]
        energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]

        # If energy is increasing too rapidly, increase compression
        if energy_trend > 0.05:
            factor = 0.95  # More compression
        elif energy_trend < -0.05:
            factor = 1.05  # Less compression
        else:
            factor = 1.0  # No change

        # Apply energy factor
        new_ratio = current_ratio * factor

        # Ensure within bounds
        return max(min(new_ratio, self.config.max_compression_ratio),
                   self.config.min_compression_ratio)

    def reset(self) -> None:
        """Reset controller state"""
        self.compression_ratio = self.config.initial_compression_ratio
        for key in self.history:
            self.history[key] = []

    def get_history(self) -> Dict[str, List]:
        """
        Get controller history.

        Returns:
            Dictionary containing controller history
        """
        return self.history

    def visualize_adaptation(self, save_path=None):
        """
        Visualize adaptation history.

        Args:
            save_path: Path to save the figure
        """
        try:
            import matplotlib.pyplot as plt

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Plot compression ratio
            x = range(len(self.history['compression_ratios']))
            ax1.plot(x, self.history['compression_ratios'], 'b-', linewidth=2)
            ax1.set_ylabel('Compression Ratio')
            ax1.set_title('Cognitive Controller Adaptation')
            ax1.grid(True)

            # Plot relevance scores
            ax2.plot(x, self.history['relevance_scores'], 'g-', linewidth=2)
            ax2.set_ylabel('Relevance Score')
            ax2.set_xlabel('Time Step')
            ax2.grid(True)

            # Add SNR if available
            if self.history['snr_values']:
                ax3 = ax1.twinx()
                ax3.plot(x[-len(self.history['snr_values']):],
                         self.history['snr_values'], 'r--', alpha=0.7)
                ax3.set_ylabel('SNR (dB)', color='r')

            # Add energy usage if available
            if self.history['energy_usage']:
                ax4 = ax2.twinx()
                ax4.plot(x[-len(self.history['energy_usage']):],
                         self.history['energy_usage'], 'm--', alpha=0.7)
                ax4.set_ylabel('Energy Usage', color='m')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)

            return fig
        except ImportError:
            print("Matplotlib not available for visualization")
            return None


class AdaptiveRISController:
    """
    Adaptive controller for RIS element configuration based on
    channel conditions and semantic performance.
    """

    def __init__(self,
                 num_elements: int = 256,
                 initial_gain_db: float = 20,
                 min_active_elements: int = 64):
        """
        Initialize the RIS controller.

        Args:
            num_elements: Total number of RIS elements
            initial_gain_db: Initial RIS gain in dB
            min_active_elements: Minimum number of active elements
        """
        self.num_elements = num_elements
        self.initial_gain_db = initial_gain_db
        self.min_active_elements = min_active_elements

        # Initialize active elements (all active initially)
        self.active_elements = num_elements

        # Initialize phase configuration (random phases)
        self.phase_config = torch.rand(num_elements) * 2 * np.pi

        # History
        self.history = {
            'active_elements': [num_elements],
            'gain_db': [initial_gain_db],
            'semantic_accuracy': []
        }

    def update(self,
               semantic_accuracy: float,
               snr_db: float,
               channel_type: str) -> Tuple[int, torch.Tensor]:
        """
        Update RIS configuration based on performance.

        Args:
            semantic_accuracy: Current semantic accuracy
            snr_db: Current SNR in dB
            channel_type: Type of channel

        Returns:
            Tuple containing number of active elements and phase configuration
        """
        # Store current accuracy
        self.history['semantic_accuracy'].append(semantic_accuracy)

        # Adapt number of active elements based on semantic accuracy
        if semantic_accuracy > 0.95:
            # Excellent accuracy - can reduce elements to save energy
            target_elements = max(self.min_active_elements,
                                  int(self.active_elements * 0.9))
        elif semantic_accuracy > 0.8:
            # Good accuracy - maintain current configuration
            target_elements = self.active_elements
        else:
            # Poor accuracy - increase elements
            target_elements = min(self.num_elements,
                                  int(self.active_elements * 1.2))

        # Adjust based on channel type
        if channel_type.lower() == 'mixed' or channel_type.lower() == 'rayleigh':
            # More challenging channels need more elements
            target_elements = min(self.num_elements,
                                  int(target_elements * 1.1))

        # Adjust based on SNR
        if snr_db < -10:
            # Very low SNR - use more elements
            target_elements = min(self.num_elements,
                                  int(target_elements * 1.2))
        elif snr_db > 20:
            # Very high SNR - can use fewer elements
            target_elements = max(self.min_active_elements,
                                  int(target_elements * 0.9))

        # Update active elements
        self.active_elements = target_elements

        # Update phase configuration (only for active elements)
        # In a real system, this would be based on channel estimation
        # For simulation, we'll just use a heuristic approach
        if channel_type.lower() == 'doppler':
            # For Doppler, adjust phases to compensate for frequency shift
            self.phase_config[:self.active_elements] = torch.rand(self.active_elements) * np.pi
        else:
            # For other channel types, use random phases within a limited range
            self.phase_config[:self.active_elements] = torch.rand(self.active_elements) * 2 * np.pi

        # Calculate effective gain
        effective_gain_db = 10 * np.log10(self.active_elements / self.num_elements) + self.initial_gain_db

        # Store history
        self.history['active_elements'].append(self.active_elements)
        self.history['gain_db'].append(effective_gain_db)

        return self.active_elements, self.phase_config

    def get_effective_gain(self) -> float:
        """
        Calculate effective RIS gain in dB.

        Returns:
            Effective RIS gain in dB
        """
        return 10 * np.log10(self.active_elements / self.num_elements) + self.initial_gain_db

    def visualize_adaptation(self, save_path=None):
        """
        Visualize RIS adaptation history.

        Args:
            save_path: Path to save the figure
        """
        try:
            import matplotlib.pyplot as plt

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Plot active elements
            x = range(len(self.history['active_elements']))
            ax1.plot(x, self.history['active_elements'], 'b-', linewidth=2)
            ax1.set_ylabel('Active RIS Elements')
            ax1.set_title('RIS Controller Adaptation')
            ax1.grid(True)

            # Plot gain
            ax2.plot(x, self.history['gain_db'], 'g-', linewidth=2)
            ax2.set_ylabel('Effective Gain (dB)')
            ax2.set_xlabel('Time Step')
            ax2.grid(True)

            # Add semantic accuracy if available
            if self.history['semantic_accuracy']:
                ax3 = ax1.twinx()
                ax3.plot(x[1:], self.history['semantic_accuracy'], 'r--', alpha=0.7)
                ax3.set_ylabel('Semantic Accuracy', color='r')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)

            return fig
        except ImportError:
            print("Matplotlib not available for visualization")
            return None