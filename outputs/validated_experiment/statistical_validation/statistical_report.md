# Statistical Validation Report

- Confidence Level: 95%
- Alpha: 0.05
- Min Sample Size: 30
- CI Method: t-based

## Summary Statistics by Configuration

### Base_AWGN_qpsk

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.729833 | [0.705974, 0.753692] | 0.209992 | 300 | 3.33e-78 |
| ber | 0.270167 | [0.246308, 0.294026] | 0.209992 | 300 | 3.33e-78 |
| energy_per_bit | 2.14823 | [2.06632, 2.23014] | 0.720947 | 300 | 7.54e-08 |
| snr_effective | 3.06381 | [2.22774, 3.89987] | 7.35855 | 300 | 5.81e-18 |

### Imperfect_RIS_qpsk

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.728509 | [0.704456, 0.752562] | 0.211702 | 300 | 4.08e-75 |
| ber | 0.271491 | [0.247438, 0.295544] | 0.211702 | 300 | 4.08e-75 |
| energy_per_bit | 2.15798 | [2.07432, 2.24164] | 0.736309 | 300 | 8.29e-08 |
| snr_effective | 3.05023 | [2.21841, 3.88204] | 7.32112 | 300 | 1.66e-17 |

### Doppler_AWGN_qpsk

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.540009 | [0.524888, 0.55513] | 0.133088 | 300 | 3e-30 |
| ber | 0.459991 | [0.44487, 0.475112] | 0.133088 | 300 | 3e-30 |
| energy_per_bit | 2.75625 | [2.69489, 2.81761] | 0.540068 | 300 | 7.66e-07 |
| snr_effective | -1.8805 | [-2.32227, -1.43872] | 3.88824 | 300 | 1.37e-65 |

### Mixed_Channel_qpsk

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.555437 | [0.537899, 0.572975] | 0.154359 | 300 | 5.02e-22 |
| ber | 0.444563 | [0.427025, 0.462101] | 0.154359 | 300 | 5.02e-22 |
| energy_per_bit | 2.71688 | [2.64807, 2.78569] | 0.60565 | 300 | 0.00107 |
| snr_effective | -1.49351 | [-2.00315, -0.983864] | 4.48556 | 300 | 3.57e-56 |

### Base_AWGN_qam16

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.682578 | [0.674862, 0.690293] | 0.0679037 | 300 | 1.15e-07 |
| ber | 0.317423 | [0.309707, 0.325138] | 0.0679037 | 300 | 1.15e-07 |
| energy_per_bit | 1.05192 | [1.03511, 1.06874] | 0.147962 | 300 | 3.08e-07 |
| snr_effective | 8.39947 | [7.4732, 9.32574] | 8.15245 | 300 | 3.57e-06 |

### Imperfect_RIS_qam16

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.681415 | [0.673634, 0.689196] | 0.0684849 | 300 | 4.11e-08 |
| ber | 0.318585 | [0.310804, 0.326366] | 0.0684849 | 300 | 4.11e-08 |
| energy_per_bit | 1.05401 | [1.03702, 1.07099] | 0.149495 | 300 | 1.71e-07 |
| snr_effective | 8.36641 | [7.44744, 9.28539] | 8.08823 | 300 | 5.7e-06 |

### Doppler_AWGN_qam16

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.549426 | [0.539219, 0.559633] | 0.0898328 | 300 | 2.87e-09 |
| ber | 0.450574 | [0.440367, 0.460781] | 0.0898328 | 300 | 2.87e-09 |
| energy_per_bit | 1.3215 | [1.29739, 1.34561] | 0.212202 | 300 | 0.64 |
| snr_effective | -0.169881 | [-0.8494, 0.509638] | 5.98071 | 300 | 6.83e-40 |

### Mixed_Channel_qam16

| Metric | Mean | 95% CI | Std | n | Normal p |
|---|---:|:---|---:|---:|---:|
| semantic_accuracy | 0.566163 | [0.555313, 0.577012] | 0.0954945 | 300 | 2.46e-12 |
| ber | 0.433838 | [0.422988, 0.444687] | 0.0954945 | 300 | 2.46e-12 |
| energy_per_bit | 1.28737 | [1.26126, 1.31347] | 0.229751 | 300 | 0.166 |
| snr_effective | 0.741725 | [-0.00618381, 1.48963] | 6.58263 | 300 | 3.51e-31 |

## Pairwise Comparisons (Holm–Bonferroni adjusted)

| Metric | A | B | Test | Mean(A) | Mean(B) | Δ | d | p (raw) | p (Holm) | Sig |
|---|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| ber | Doppler_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.459991 | 0.450574 | 0.00941667 | 0.0829 | 0.00149 | 6.71e-94 | ✔ |
| ber | Mixed_Channel_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.444563 | 0.317423 | 0.127141 | 1.07 | 3.97e-44 | 2.26e-91 | ✔ |
| ber | Mixed_Channel_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.444563 | 0.318585 | 0.125978 | 1.06 | 1.11e-43 | 4.16e-88 | ✔ |
| ber | Doppler_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.459991 | 0.318585 | 0.141406 | 1.34 | 9.96e-54 | 2e-60 | ✔ |
| ber | Doppler_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 0.450574 | 0.433838 | 0.0167367 | 0.181 | 0.044 | 2.06e-51 | ✔ |
| ber | Base_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.270167 | 0.433838 | -0.163671 | -1 | 9.28e-20 | 2.06e-50 | ✔ |
| ber | Doppler_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 0.459991 | 0.444563 | 0.0154275 | 0.107 | 0.495 | 2.06e-50 | ✔ |
| ber | Mixed_Channel_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.444563 | 0.450574 | -0.00601083 | -0.0476 | 0.0282 | 3.94e-46 | ✔ |
| ber | Mixed_Channel_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.444563 | 0.433838 | 0.0107258 | 0.0836 | 7.37e-05 | 2.46e-23 | ✔ |
| ber | Imperfect_RIS_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 0.271491 | 0.459991 | -0.1885 | -1.07 | 1.62e-23 | 3.16e-23 | ✔ |
| ber | Base_AWGN_qpsk | Imperfect_RIS_qpsk | Mann–Whitney U | 0.270167 | 0.271491 | -0.00132417 | -0.00628 | 0.936 | 3.16e-23 | ✔ |
| ber | Doppler_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.459991 | 0.317423 | 0.142568 | 1.35 | 3.84e-54 | 5.16e-23 | ✔ |
| ber | Base_AWGN_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 0.270167 | 0.459991 | -0.189824 | -1.08 | 1.58e-24 | 1.33e-21 | ✔ |
| ber | Imperfect_RIS_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 0.271491 | 0.444563 | -0.173072 | -0.934 | 1.05e-20 | 1.33e-21 | ✔ |
| ber | Base_AWGN_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 0.317423 | 0.450574 | -0.133152 | -1.67 | 9.93e-53 | 3.15e-19 | ✔ |
| ber | Base_AWGN_qam16 | Imperfect_RIS_qam16 | Mann–Whitney U | 0.317423 | 0.318585 | -0.0011625 | -0.017 | 0.819 | 8.74e-11 | ✔ |
| ber | Imperfect_RIS_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 0.318585 | 0.450574 | -0.131989 | -1.65 | 8.24e-52 | 2.68e-07 | ✔ |
| ber | Imperfect_RIS_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 0.318585 | 0.433838 | -0.115253 | -1.39 | 1.58e-41 | 4.85e-06 | ✔ |
| ber | Imperfect_RIS_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.271491 | 0.450574 | -0.179083 | -1.1 | 4.46e-22 | 6.63e-06 | ✔ |
| ber | Base_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.270167 | 0.318585 | -0.0484183 | -0.31 | 0.00114 | 6.63e-06 | ✔ |
| ber | Base_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.270167 | 0.450574 | -0.180408 | -1.12 | 7.39e-23 | 0.000811 | ✔ |
| ber | Imperfect_RIS_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.271491 | 0.433838 | -0.162347 | -0.989 | 3.13e-19 | 0.000811 | ✔ |
| ber | Base_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 0.270167 | 0.444563 | -0.174397 | -0.946 | 1.31e-21 | 0.0118 | ✔ |
| ber | Imperfect_RIS_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.271491 | 0.317423 | -0.0459317 | -0.292 | 0.00156 | 0.0118 | ✔ |
| ber | Base_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 0.317423 | 0.433838 | -0.116415 | -1.41 | 2.8e-42 | 1 | ✘ |
| ber | Doppler_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.459991 | 0.433838 | 0.0261533 | 0.226 | 5.52e-07 | 1 | ✘ |
| ber | Imperfect_RIS_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.271491 | 0.318585 | -0.0470942 | -0.299 | 0.0013 | 1 | ✘ |
| ber | Base_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.270167 | 0.317423 | -0.0472558 | -0.303 | 0.00148 | 1 | ✘ |
| energy_per_bit | Doppler_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 2.75625 | 1.28737 | 1.46888 | 3.54 | 1.13e-92 | 1.52e-97 | ✔ |
| energy_per_bit | Mixed_Channel_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 2.71688 | 1.05192 | 1.66495 | 3.78 | 1.85e-98 | 4.8e-97 | ✔ |
| energy_per_bit | Doppler_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 2.75625 | 1.3215 | 1.43475 | 3.5 | 1.15e-91 | 1.61e-93 | ✔ |
| energy_per_bit | Base_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 1.05192 | 1.28737 | -0.235443 | -1.22 | 2.57e-37 | 6.05e-74 | ✔ |
| energy_per_bit | Imperfect_RIS_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 1.05401 | 1.3215 | -0.267491 | -1.46 | 3.58e-47 | 3.18e-68 | ✔ |
| energy_per_bit | Doppler_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 2.75625 | 1.05401 | 1.70225 | 4.3 | 6.3e-99 | 2.52e-64 | ✔ |
| energy_per_bit | Base_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 2.14823 | 1.05192 | 1.09631 | 2.11 | 2.64e-95 | 1.07e-52 | ✔ |
| energy_per_bit | Imperfect_RIS_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 2.15798 | 1.05401 | 1.10397 | 2.08 | 7.67e-95 | 1.07e-52 | ✔ |
| energy_per_bit | Base_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 2.14823 | 1.05401 | 1.09422 | 2.1 | 6.56e-95 | 9.53e-43 | ✔ |
| energy_per_bit | Imperfect_RIS_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 2.15798 | 1.3215 | 0.836483 | 1.54 | 3.25e-61 | 9.53e-43 | ✔ |
| energy_per_bit | Doppler_AWGN_qam16 | Mixed_Channel_qam16 | Welch t-test | 1.3215 | 1.28737 | 0.0341302 | 0.154 | 0.0592 | 8.78e-41 | ✔ |
| energy_per_bit | Base_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 2.14823 | 1.28737 | 0.860863 | 1.61 | 1.57e-65 | 3.32e-40 | ✔ |
| energy_per_bit | Doppler_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 2.75625 | 2.71688 | 0.0393737 | 0.0686 | 0.459 | 3.32e-40 | ✔ |
| energy_per_bit | Mixed_Channel_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 2.71688 | 1.3215 | 1.39538 | 3.07 | 8.46e-88 | 7.85e-36 | ✔ |
| energy_per_bit | Mixed_Channel_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 2.71688 | 1.28737 | 1.42951 | 3.12 | 2.31e-89 | 1.81e-20 | ✔ |
| energy_per_bit | Imperfect_RIS_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 2.15798 | 2.75625 | -0.598271 | -0.927 | 7.76e-23 | 2.09e-20 | ✔ |
| energy_per_bit | Base_AWGN_qpsk | Imperfect_RIS_qpsk | Mann–Whitney U | 2.14823 | 2.15798 | -0.00975075 | -0.0134 | 0.928 | 2.09e-20 | ✔ |
| energy_per_bit | Doppler_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 2.75625 | 1.05192 | 1.70433 | 4.3 | 5.43e-99 | 5.23e-20 | ✔ |
| energy_per_bit | Base_AWGN_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 1.05192 | 1.3215 | -0.269573 | -1.47 | 4.71e-48 | 3.24e-19 | ✔ |
| energy_per_bit | Base_AWGN_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 2.14823 | 2.75625 | -0.608022 | -0.955 | 6.45e-24 | 1.3e-18 | ✔ |
| energy_per_bit | Imperfect_RIS_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 2.15798 | 2.71688 | -0.558897 | -0.829 | 6.38e-20 | 1.3e-18 | ✔ |
| energy_per_bit | Base_AWGN_qam16 | Imperfect_RIS_qam16 | Mann–Whitney U | 1.05192 | 1.05401 | -0.00208232 | -0.014 | 0.896 | 4.85e-06 | ✔ |
| energy_per_bit | Imperfect_RIS_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 2.15798 | 1.05192 | 1.10606 | 2.08 | 2.92e-95 | 0.0117 | ✔ |
| energy_per_bit | Base_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 2.14823 | 2.71688 | -0.568648 | -0.854 | 8.72e-21 | 0.0117 | ✔ |
| energy_per_bit | Mixed_Channel_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 2.71688 | 1.05401 | 1.66287 | 3.77 | 2.19e-98 | 1 | ✘ |
| energy_per_bit | Imperfect_RIS_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 2.15798 | 1.28737 | 0.870614 | 1.6 | 2.62e-65 | 1 | ✘ |
| energy_per_bit | Base_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 2.14823 | 1.3215 | 0.826733 | 1.56 | 1.43e-61 | 1 | ✘ |
| energy_per_bit | Imperfect_RIS_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 1.05401 | 1.28737 | -0.233361 | -1.2 | 8.72e-37 | 1 | ✘ |
| semantic_accuracy | Doppler_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.540009 | 0.681415 | -0.141406 | -1.34 | 9.96e-54 | 1.44e-93 | ✔ |
| semantic_accuracy | Mixed_Channel_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.555437 | 0.682578 | -0.127141 | -1.07 | 3.97e-44 | 2.18e-90 | ✔ |
| semantic_accuracy | Mixed_Channel_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.555437 | 0.681415 | -0.125978 | -1.06 | 1.11e-43 | 1.44e-86 | ✔ |
| semantic_accuracy | Doppler_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.540009 | 0.566163 | -0.0261533 | -0.226 | 5.52e-07 | 3.93e-64 | ✔ |
| semantic_accuracy | Base_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.729833 | 0.566163 | 0.163671 | 1 | 9.28e-20 | 6.16e-41 | ✔ |
| semantic_accuracy | Doppler_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 0.540009 | 0.555437 | -0.0154275 | -0.107 | 0.495 | 6.16e-41 | ✔ |
| semantic_accuracy | Doppler_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 0.549426 | 0.566163 | -0.0167367 | -0.181 | 0.044 | 8.78e-41 | ✔ |
| semantic_accuracy | Mixed_Channel_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.555437 | 0.549426 | 0.00601083 | 0.0476 | 0.0282 | 2.57e-36 | ✔ |
| semantic_accuracy | Base_AWGN_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 0.682578 | 0.549426 | 0.133152 | 1.67 | 9.93e-53 | 1.85e-20 | ✔ |
| semantic_accuracy | Base_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 0.729833 | 0.555437 | 0.174397 | 0.946 | 1.31e-21 | 1.57e-19 | ✔ |
| semantic_accuracy | Imperfect_RIS_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.728509 | 0.682578 | 0.0459317 | 0.292 | 0.00156 | 1.57e-19 | ✔ |
| semantic_accuracy | Doppler_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.540009 | 0.549426 | -0.00941667 | -0.0829 | 0.00149 | 3.19e-19 | ✔ |
| semantic_accuracy | Base_AWGN_qam16 | Imperfect_RIS_qam16 | Mann–Whitney U | 0.682578 | 0.681415 | 0.0011625 | 0.017 | 0.819 | 3.24e-19 | ✔ |
| semantic_accuracy | Imperfect_RIS_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.728509 | 0.681415 | 0.0470942 | 0.299 | 0.0013 | 4.07e-18 | ✔ |
| semantic_accuracy | Base_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.729833 | 0.682578 | 0.0472558 | 0.303 | 0.00148 | 4.07e-18 | ✔ |
| semantic_accuracy | Base_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 0.682578 | 0.566163 | 0.116415 | 1.41 | 2.8e-42 | 4.85e-06 | ✔ |
| semantic_accuracy | Imperfect_RIS_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 0.681415 | 0.549426 | 0.131989 | 1.65 | 8.24e-52 | 0.00222 | ✔ |
| semantic_accuracy | Base_AWGN_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 0.729833 | 0.540009 | 0.189824 | 1.08 | 1.58e-24 | 0.0114 | ✔ |
| semantic_accuracy | Imperfect_RIS_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 0.728509 | 0.555437 | 0.173073 | 0.934 | 1.05e-20 | 0.0114 | ✔ |
| semantic_accuracy | Imperfect_RIS_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.728509 | 0.549426 | 0.179083 | 1.1 | 4.46e-22 | 0.0118 | ✔ |
| semantic_accuracy | Base_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 0.729833 | 0.681415 | 0.0484183 | 0.31 | 0.00114 | 0.0118 | ✔ |
| semantic_accuracy | Imperfect_RIS_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 0.681415 | 0.566163 | 0.115253 | 1.39 | 1.58e-41 | 0.0152 | ✔ |
| semantic_accuracy | Base_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 0.729833 | 0.549426 | 0.180408 | 1.12 | 7.39e-23 | 0.141 | ✘ |
| semantic_accuracy | Imperfect_RIS_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.728509 | 0.566163 | 0.162347 | 0.989 | 3.13e-19 | 0.141 | ✘ |
| semantic_accuracy | Doppler_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 0.540009 | 0.682578 | -0.142568 | -1.35 | 3.84e-54 | 1 | ✘ |
| semantic_accuracy | Imperfect_RIS_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 0.728509 | 0.540009 | 0.1885 | 1.07 | 1.62e-23 | 1 | ✘ |
| semantic_accuracy | Mixed_Channel_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 0.555437 | 0.566163 | -0.0107258 | -0.0836 | 7.37e-05 | 1 | ✘ |
| semantic_accuracy | Base_AWGN_qpsk | Imperfect_RIS_qpsk | Mann–Whitney U | 0.729833 | 0.728509 | 0.00132417 | 0.00628 | 0.936 | 1 | ✘ |
| snr_effective | Doppler_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | -1.8805 | 0.741725 | -2.62222 | -0.485 | 2.68e-08 | 1.7e-97 | ✔ |
| snr_effective | Mixed_Channel_qpsk | Base_AWGN_qam16 | Mann–Whitney U | -1.49351 | 8.39947 | -9.89298 | -1.5 | 1.22e-69 | 5.47e-97 | ✔ |
| snr_effective | Doppler_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | -1.8805 | 8.39947 | -10.28 | -1.61 | 2.16e-75 | 6.35e-94 | ✔ |
| snr_effective | Base_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 8.39947 | 0.741725 | 7.65775 | 1.03 | 3.99e-42 | 6.05e-74 | ✔ |
| snr_effective | Imperfect_RIS_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 8.36641 | -0.169881 | 8.53629 | 1.2 | 8.95e-53 | 3.24e-68 | ✔ |
| snr_effective | Doppler_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | -1.8805 | -0.169881 | -1.71062 | -0.339 | 0.00037 | 4.22e-60 | ✔ |
| snr_effective | Imperfect_RIS_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 3.05023 | 8.36641 | -5.31619 | -0.689 | 2.16e-20 | 2.69e-52 | ✔ |
| snr_effective | Base_AWGN_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 3.06381 | 8.39947 | -5.33567 | -0.687 | 2.17e-20 | 2.69e-52 | ✔ |
| snr_effective | Imperfect_RIS_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | 8.36641 | 0.741725 | 7.62469 | 1.03 | 4.18e-42 | 1.98e-51 | ✔ |
| snr_effective | Base_AWGN_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 3.06381 | -0.169881 | 3.23369 | 0.482 | 7.28e-12 | 2.58e-51 | ✔ |
| snr_effective | Imperfect_RIS_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 3.05023 | 0.741725 | 2.3085 | 0.332 | 5.47e-07 | 2.58e-51 | ✔ |
| snr_effective | Mixed_Channel_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | -1.49351 | 8.36641 | -9.85992 | -1.51 | 1.3e-69 | 5.66e-47 | ✔ |
| snr_effective | Base_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | 3.06381 | 8.36641 | -5.30261 | -0.686 | 2.33e-20 | 2.54e-42 | ✔ |
| snr_effective | Imperfect_RIS_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | 3.05023 | -0.169881 | 3.22011 | 0.482 | 7.33e-12 | 2.54e-42 | ✔ |
| snr_effective | Base_AWGN_qam16 | Imperfect_RIS_qam16 | Mann–Whitney U | 8.39947 | 8.36641 | 0.0330598 | 0.00407 | 0.93 | 2.46e-23 | ✔ |
| snr_effective | Base_AWGN_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 3.06381 | -1.8805 | 4.94431 | 0.84 | 1.23e-24 | 3.08e-22 | ✔ |
| snr_effective | Imperfect_RIS_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 3.05023 | -1.49351 | 4.54373 | 0.748 | 1.09e-21 | 3.08e-22 | ✔ |
| snr_effective | Doppler_AWGN_qpsk | Imperfect_RIS_qam16 | Mann–Whitney U | -1.8805 | 8.36641 | -10.2469 | -1.61 | 2.2e-75 | 5.43e-22 | ✔ |
| snr_effective | Base_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | 3.06381 | -1.49351 | 4.55731 | 0.748 | 1.01e-21 | 7.58e-21 | ✔ |
| snr_effective | Imperfect_RIS_qpsk | Base_AWGN_qam16 | Mann–Whitney U | 3.05023 | 8.39947 | -5.34925 | -0.69 | 1.97e-20 | 7.58e-21 | ✔ |
| snr_effective | Mixed_Channel_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | -1.49351 | 0.741725 | -2.23523 | -0.397 | 6.32e-07 | 3.24e-19 | ✔ |
| snr_effective | Base_AWGN_qam16 | Doppler_AWGN_qam16 | Mann–Whitney U | 8.39947 | -0.169881 | 8.56935 | 1.2 | 8.26e-53 | 8.74e-11 | ✔ |
| snr_effective | Imperfect_RIS_qpsk | Doppler_AWGN_qpsk | Mann–Whitney U | 3.05023 | -1.8805 | 4.93073 | 0.841 | 1.27e-24 | 0.0118 | ✔ |
| snr_effective | Base_AWGN_qpsk | Imperfect_RIS_qpsk | Mann–Whitney U | 3.06381 | 3.05023 | 0.0135789 | 0.00185 | 0.949 | 0.0118 | ✔ |
| snr_effective | Base_AWGN_qpsk | Mixed_Channel_qam16 | Mann–Whitney U | 3.06381 | 0.741725 | 2.32208 | 0.333 | 5.39e-07 | 0.176 | ✘ |
| snr_effective | Doppler_AWGN_qpsk | Mixed_Channel_qpsk | Mann–Whitney U | -1.8805 | -1.49351 | -0.386992 | -0.0922 | 0.532 | 0.176 | ✘ |
| snr_effective | Doppler_AWGN_qam16 | Mixed_Channel_qam16 | Mann–Whitney U | -0.169881 | 0.741725 | -0.911606 | -0.145 | 0.0553 | 0.221 | ✘ |
| snr_effective | Mixed_Channel_qpsk | Doppler_AWGN_qam16 | Mann–Whitney U | -1.49351 | -0.169881 | -1.32363 | -0.25 | 0.00303 | 0.237 | ✘ |

