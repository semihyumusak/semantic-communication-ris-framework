# Semantic Communication RIS Framework

An experimental framework for cognitive semantic communication over Reconfigurable Intelligent Surface (RIS)-enabled wireless links with Rate-Distortion-Relevance (RDR) optimization.

## Paper Reference

This repository accompanies the paper: **"An Experimental Information-Theoretic Framework for Cognitive Semantic Communication: A Rate–Distortion–Relevance Study over RIS-Enabled Wireless Links"** by Semih Yumuşak (University of Southampton).

## Overview

Traditional wireless systems focus on bit-level fidelity, but this framework emphasizes semantic preservation and energy efficiency in challenging channel conditions. The system demonstrates how RIS technology can enhance semantic communication performance, particularly in low-SNR environments.

### Key Results

- **93% semantic accuracy** at -10 dB SNR with RIS vs **15%** without RIS
- **Up to 65 percentage point improvements** in challenging conditions
- **Multi-user validation**: 94-99% critical user accuracy with semantic-aware allocation
- **Logarithmic scaling** behavior confirmed for RIS element optimization

## Features

- **Rate-Distortion-Relevance (RDR) Framework**: Three-dimensional optimization balancing rate, distortion, and semantic relevance
- **RIS-Enhanced Communication**: Configurable intelligent surfaces with 64-256 elements
- **Multi-User Resource Allocation**: Semantic-aware, equal, and priority-weighted strategies
- **Channel Models**: Base AWGN, Imperfect RIS, Doppler AWGN, Mixed Channel scenarios
- **Statistical Validation**: 1000 Monte Carlo trials per configuration with confidence intervals

## Installation

```bash
git clone https://github.com/semihyumusak/semantic-communication-ris-framework.git
cd semantic-communication-ris-framework
pip install -r requirements.txt
```

## Requirements

```
numpy>=1.21.0
torch>=1.9.0
matplotlib>=3.4.0
scipy>=1.7.0
pandas>=1.3.0
tqdm>=4.62.0
```

## Quick Start

### Single-User Semantic Communication

```python
from multi_user_experiment import MultiUserSemanticAllocation

# Initialize experiment
experiment = MultiUserSemanticAllocation()

# Run comprehensive validation
results = experiment.run_multi_user_experiment()

print(f"Results saved to {experiment.output_dir}")
```

### Key Experiments

1. **Multi-User Resource Allocation**:
   ```bash
   python multi_user_experiment.py
   ```

2. **Baseline RIS vs Non-RIS Comparison**:
   ```python
   # Compare semantic-RIS system against conventional approaches
   # Generates Figure 6 and Table 3 from the paper
   ```

3. **RIS Scaling Analysis**:
   ```python
   # Validate logarithmic scaling behavior
   # Tests 64, 128, and 256 RIS elements
   ```

## Repository Structure

```
semantic-communication-ris-framework/
├── multi_user_experiment.py        # Main multi-user validation (500 trials)
├── outputs/                        # Generated results and plots
│   └── multi_user_validation/
├── figures/                        # Paper figures
│   ├── multi_user_allocation_comparison.png
│   └── critical_user_performance.png
├── requirements.txt
└── README.md
```

## Key Experimental Configurations

### User Types (Multi-User Scenarios)
- **Critical Medical**: Priority 1.0, min 95% accuracy
- **Important Video**: Priority 0.6, min 85% accuracy  
- **Background Data**: Priority 0.2, min 70% accuracy

### Allocation Strategies
- **Equal Allocation**: Uniform resource distribution (baseline)
- **Semantic-Aware**: Priority + accuracy requirements
- **Priority-Weighted**: Pure priority-based allocation

### Channel Conditions
- **Base AWGN**: Ideal RIS with AWGN
- **Imperfect RIS**: Phase noise in RIS elements
- **Doppler AWGN**: Mobility effects
- **Mixed Channel**: Multipath fading + spatial correlation

## Key Results Reproduction

### Multi-User Critical Findings
- Semantic-aware allocation: **94-99% critical user accuracy**
- Equal allocation baseline: **72-98% critical user accuracy**
- System utility improvement: **15-25%** at low SNR
- Statistical significance: **p < 0.001, Cohen's d = 1.2**

### RIS Benefits
- **12 dB effective SNR improvement** with RIS
- **Logarithmic scaling** with diminishing returns beyond 128 elements
- **Energy-semantic alignment**: Better semantic performance correlates with energy efficiency

## File Outputs

The framework generates several output files:

- `multi_user_allocation_results.csv`: Complete experimental results
- `fairness_efficiency_tradeoff.csv`: Trade-off analysis
- `multi_user_analysis_report.md`: Detailed findings summary
- Visualization plots (PNG format)

## Statistical Validation

All experiments include:
- **500-1000 Monte Carlo trials** per configuration
- **95% confidence intervals**
- **Statistical significance testing** (t-tests, α = 0.05)
- **Effect size calculations** (Cohen's d)

## Citing This Work

If you use this framework in your research, please cite:

```bibtex
@article{yumusak2025semantic,
  title={An Experimental Information-Theoretic Framework for Cognitive Semantic Communication: A Rate–Distortion–Relevance Study over RIS-Enabled Wireless Links},
  author={Yumuşak, Semih},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Semih Yumuşak**  
Department of Electrical and Computer Science  
University of Southampton, United Kingdom  
Email: [contact email]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any improvements or bug fixes.

## Acknowledgments

This research contributes to the theoretical foundations necessary for practical 6G semantic communication systems where semantic relevance is jointly optimized with physical-layer constraints.
