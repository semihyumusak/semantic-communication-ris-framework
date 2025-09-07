# Semantic Communication RIS Framework

An experimental framework for cognitive semantic communication over Reconfigurable Intelligent Surface (RIS)-enabled wireless links with Rate-Distortion-Relevance (RDR) optimization.

## Paper Reference

This repository accompanies the paper: **"An Experimental Information-Theoretic Framework for Cognitive Semantic Communication: A Rate–Distortion–Relevance Study over RIS-Enabled Wireless Links"** by Semih Yumuşak (University of Southampton).

## Overview

Traditional wireless systems focus on bit-level fidelity, but this framework emphasizes semantic preservation and energy efficiency in challenging channel conditions. The system demonstrates how RIS technology can enhance semantic communication performance, particularly in low-SNR environments.

### Key Results

- **93% semantic accuracy** at -10 dB SNR with RIS vs **15%** without RIS  
- **Up to 65 percentage point improvements** in challenging conditions  
- **Multi-user validation**: 94–99% critical user accuracy with semantic-aware allocation  
- **Logarithmic scaling** behaviour confirmed for RIS element optimization  

## Features

- **Rate-Distortion-Relevance (RDR) Framework**: Balancing rate, distortion, and semantic relevance  
- **RIS-Enhanced Communication**: Configurable intelligent surfaces with 64–256 elements  
- **Multi-User Resource Allocation**: Semantic-aware, equal, and priority-weighted strategies  
- **Channel Models**: AWGN, imperfect RIS, Doppler AWGN, and mixed channels  
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

## Running All Experiments

To reproduce **all results and figures** from the paper, you need to execute the following experiment scripts manually. Each script generates outputs (CSV, plots, reports) inside the `outputs/` and `figures/` directories.

Run them in order:

```bash
python enhanced_simulation.py
python multi_user_experiment.py
python non_semantic_baseline_experiment.py
python realistic_critical_experiments.py
python statistical_validation_fix.py
```

### Script Descriptions

- **`enhanced_simulation.py`** – Single-user semantic communication with RIS scaling  
- **`multi_user_experiment.py`** – Multi-user resource allocation experiments  
- **`non_semantic_baseline_experiment.py`** – Baseline comparison without semantic/RIS enhancements  
- **`realistic_critical_experiments.py`** – Critical user scenarios and real-world relevance  
- **`statistical_validation_fix.py`** – Monte Carlo validation and statistical significance tests  

## File Outputs

- `multi_user_allocation_results.csv` – Multi-user experiment results  
- `fairness_efficiency_tradeoff.csv` – Fairness vs efficiency analysis  
- `multi_user_analysis_report.md` – Detailed multi-user findings  
- Figures in PNG format (allocation comparison, scaling curves, etc.)  

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
Email: semihyumusak@yahoo.com
