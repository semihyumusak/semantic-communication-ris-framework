# Cognitive Semantic Communication over RIS-Enabled Links
## Summary of Experimental Results

### Execution Time
- standard_experiment: 8.95 seconds
- rdr_analysis: 2.77 seconds
- baseline_comparison: 0.51 seconds

### Key Findings
1. **Impact of RIS Elements**: Increasing RIS elements improves semantic accuracy logarithmically.
2. **Channel Robustness**: Semantic communication shows greater resilience to challenging channel conditions compared to bit-level metrics.
3. **Adaptive Control**: The cognitive controller successfully adapts compression ratio based on channel conditions.
4. **Energy Efficiency**: RIS-enabled links significantly improve energy per semantic bit, especially at low SNRs.

### Figures
Key visualizations are available in the `figures` directory:
- Accuracy vs SNR curves for different channel conditions
- Energy efficiency comparisons
- RDR surface plots showing the trade-off between rate, distortion, and relevance
- Baseline comparisons

### Tables
Detailed results are available in the `tables` directory:
- Detailed performance metrics for each test case
- Scaling results with different RIS configurations
- Baseline comparison data

### RDR Data
Raw data for Rate-Distortion-Relevance analysis is available in the `rdr_data` directory.
