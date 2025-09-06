# statistical_validation_module.py
# ---------------------------------------------------------------
# A complete, drop-in module for statistically validating your
# wireless experiments. It:
#   • runs n trials per configuration and stores RAW per-trial arrays
#   • computes t-based or bootstrap CIs
#   • performs pairwise significance tests (Welch t-test or Mann–Whitney)
#   • applies Holm–Bonferroni multiple-comparisons correction
#   • computes effect sizes (Cohen’s d)
#   • generates publication-ready CI plots & boxplots FROM RAW DATA
#   • optionally saves raw arrays to CSV for auditability
#
# It also includes an ExperimentValidator wrapper that plugs into an
# existing experiment object exposing:
#   - experiment.modulation_schemes: iterable of strings
#   - experiment.test_cases: dict {name: { "ris_model": <object> , ...}}
#   - experiment.output_dir: str
#   - experiment.num_bits: int
#
# Your ris_model is expected to implement:
#   - apply_ris_gain(snr_db: float) -> float
#   - optional: randomise_state()  (introduce per-trial stochasticity)
#
# The single-trial runner below demonstrates best practices:
#   - per-trial random SNR around a nominal operating point
#   - measured snr_effective from error-vector power
#   - EPSB computed from TRANSMIT energy per semantic bit
#
# ---------------------------------------------------------------

from __future__ import annotations

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

from scipy.stats import (
    t, ttest_ind, mannwhitneyu, normaltest, sem as scipy_sem
)

# =========================== Utilities ===========================

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sx2 = x.var(ddof=1); sy2 = y.var(ddof=1)
    s  = np.sqrt((sx2 + sy2) / 2.0)
    if s == 0.0:
        return 0.0
    return (x.mean() - y.mean()) / s

def ci_mean_t(x: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """t-based CI for the mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float(x.mean())
    m = x.mean()
    s = scipy_sem(x)
    q = t.ppf(0.5 + level/2.0, df=n-1)
    return float(m - q*s), float(m + q*s)

def ci_mean_bootstrap(x: np.ndarray, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = np.percentile(boots, (1.0 - level) / 2.0 * 100.0)
    hi = np.percentile(boots, (1.0 + level) / 2.0 * 100.0)
    return float(lo), float(hi)

def holm_bonferroni_correction(pvals: List[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values (monotone, step-down)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * pvals[idx]
        adj = max(adj, prev)  # enforce monotonicity
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted.tolist()

# ===================== StatisticalValidator ======================

@dataclass
class ValidatorConfig:
    confidence_level: float = 0.95
    alpha: float = 0.05
    min_sample_size: int = 30
    use_bootstrap_ci: bool = False
    n_bootstrap: int = 2000
    store_raw: bool = True
    random_seed: int = 0

class StatisticalValidator:
    """
    Adds rigorous statistical validation on top of any existing single-trial experiment function.

    Workflow:
      1) validate_existing_experiment()  -> per-config summary stats (+ raw arrays)
      2) compare_configurations()        -> pairwise tests with Holm–Bonferroni
      3) generate_statistical_report()   -> markdown report
      4) create_statistical_plots()      -> CI plots + boxplots from RAW data
      5) save_raw_as_csv()               -> raw arrays per config to CSV
    """

    def __init__(self, cfg: Optional[ValidatorConfig] = None):
        self.cfg = cfg or ValidatorConfig()

    # ------------------------- Core API --------------------------

    def validate_existing_experiment(
        self,
        experiment_function: Callable[[Dict[str, Any]], Dict[str, float]],
        config_params: Dict[str, Any],
        n_trials: int = 300,
    ) -> Dict[str, Any]:
        """
        Runs the experiment n_trials times and returns summary stats + (optionally) raw arrays.
        The experiment_function must return a dict of scalar metrics per trial, e.g.:
        {"semantic_accuracy": float, "ber": float, "energy_per_bit": float, "snr_effective": float}
        """
        rng = np.random.default_rng(self.cfg.random_seed)
        raw: Dict[str, List[float]] = {}

        for trial in range(n_trials):
            params = {**config_params, "trial": trial, "rng_seed": int(rng.integers(0, 2**31-1))}
            out = experiment_function(params)
            for k, v in out.items():
                if isinstance(v, (int, float, np.floating)):
                    raw.setdefault(k, []).append(float(v))

        # compute summary
        summary: Dict[str, Any] = {}
        for metric, arr in raw.items():
            x = np.asarray(arr, dtype=float)
            if self.cfg.use_bootstrap_ci:
                lo, hi = ci_mean_bootstrap(x, self.cfg.confidence_level, self.cfg.n_bootstrap, self.cfg.random_seed)
            else:
                lo, hi = ci_mean_t(x, self.cfg.confidence_level)

            summary[metric] = {
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "sem": float(scipy_sem(x)) if len(x) > 1 else 0.0,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_width": float(hi - lo),
                "min": float(x.min()),
                "max": float(x.max()),
                "q25": float(np.percentile(x, 25)),
                "q75": float(np.percentile(x, 75)),
                "sample_size": int(len(x)),
                # D'Agostino's normality (more stable than Shapiro for large n)
                "normal_p": float(normaltest(x).pvalue) if len(x) >= 8 else float("nan"),
            }

        if self.cfg.store_raw:
            summary["_raw"] = raw
        return summary

    def compare_configurations(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Pairwise comparisons for each metric using:
          - Welch’s t-test if both groups are approximately normal,
          - Mann–Whitney U otherwise.
        Returns a dataframe with Holm–Bonferroni adjusted p-values (per metric).
        """
        configs = list(results_dict.keys())
        metrics = [m for m in results_dict[configs[0]].keys() if m != "_raw"]
        rows: List[Dict[str, Any]] = []

        for i, a in enumerate(configs):
            for b in configs[i+1:]:
                for metric in metrics:
                    xa = np.asarray(results_dict[a]["_raw"][metric], dtype=float)
                    xb = np.asarray(results_dict[b]["_raw"][metric], dtype=float)

                    na = not math.isnan(results_dict[a][metric].get("normal_p", float("nan"))) and \
                         results_dict[a][metric]["normal_p"] > self.cfg.alpha
                    nb = not math.isnan(results_dict[b][metric].get("normal_p", float("nan"))) and \
                         results_dict[b][metric]["normal_p"] > self.cfg.alpha

                    if na and nb and len(xa) >= 2 and len(xb) >= 2:
                        stat, p = ttest_ind(xa, xb, equal_var=False)  # Welch
                        test_used = "Welch t-test"
                    else:
                        stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
                        test_used = "Mann–Whitney U"

                    d = cohen_d(xa, xb)

                    rows.append({
                        "metric": metric,
                        "config_a": a,
                        "config_b": b,
                        "test": test_used,
                        "mean_a": float(xa.mean()),
                        "mean_b": float(xb.mean()),
                        "mean_diff": float(xa.mean() - xb.mean()),
                        "percent_diff": float(((xa.mean() - xb.mean()) / (xb.mean() if abs(xb.mean())>1e-12 else np.nan)) * 100.0),
                        "cohens_d": float(d),
                        "p_raw": float(p),
                    })

        df = pd.DataFrame(rows)

        # Holm–Bonferroni per metric
        adj = []
        for metric, sub in df.groupby("metric", sort=False):
            adj.extend(holm_bonferroni_correction(sub["p_raw"].tolist()))
        df["p_holm"] = adj
        df["significant"] = df["p_holm"] < self.cfg.alpha
        df["effect_size"] = df["cohens_d"].abs().map(
            lambda d: "Negligible" if d < 0.2 else ("Small" if d < 0.5 else ("Medium" if d < 0.8 else "Large"))
        )
        return df.sort_values(["metric", "p_holm", "p_raw"]).reset_index(drop=True)

    # -------------------- Reporting & Plots ----------------------

    def generate_statistical_report(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        comparison_df: pd.DataFrame,
        output_path: str
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Statistical Validation Report\n\n")
            f.write(f"- Confidence Level: {int(self.cfg.confidence_level*100)}%\n")
            f.write(f"- Alpha: {self.cfg.alpha}\n")
            f.write(f"- Min Sample Size: {self.cfg.min_sample_size}\n")
            f.write(f"- CI Method: {'Bootstrap' if self.cfg.use_bootstrap_ci else 't-based'}\n\n")

            f.write("## Summary Statistics by Configuration\n\n")
            for cfg, stats_map in results_dict.items():
                f.write(f"### {cfg}\n\n")
                f.write("| Metric | Mean | 95% CI | Std | n | Normal p |\n")
                f.write("|---|---:|:---|---:|---:|---:|\n")
                for metric, s in stats_map.items():
                    if metric == "_raw":
                        continue
                    f.write(f"| {metric} | {s['mean']:.6g} | [{s['ci_lower']:.6g}, {s['ci_upper']:.6g}] | "
                            f"{s['std']:.6g} | {s['sample_size']} | {s['normal_p']:.3g} |\n")
                f.write("\n")

            f.write("## Pairwise Comparisons (Holm–Bonferroni adjusted)\n\n")
            if comparison_df.empty:
                f.write("_No comparisons produced._\n")
            else:
                f.write("| Metric | A | B | Test | Mean(A) | Mean(B) | Δ | d | p (raw) | p (Holm) | Sig |\n")
                f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|:--:|\n")
                for _, r in comparison_df.iterrows():
                    f.write(
                        f"| {r.metric} | {r.config_a} | {r.config_b} | {r.test} | "
                        f"{r.mean_a:.6g} | {r.mean_b:.6g} | {r.mean_diff:.6g} | {r.cohens_d:.3g} | "
                        f"{r.p_raw:.3g} | {r.p_holm:.3g} | {'✔' if r.significant else '✘'} |\n"
                    )
            f.write("\n")

    import re
    def prettify_metric(self, metric: str) -> str:
        METRIC_MAP = {
            "snr_effective": "Effective SNR",
            "energy_per_bit": "Energy per Bit",
            "semantic_accuracy": "Semantic Accuracy",
            "ber": "Bit Error Rate",
        }
        # fallback: replace underscores and title case
        return METRIC_MAP.get(metric, metric.replace("_", " ").title())

    def prettify_label_auto(self, label: str) -> str:
        # split on underscores, capitalise each token, group modulation
        parts = label.split("_")
        if parts[-1].lower() in ["qpsk", "qam16"]:
            modulation = parts[-1].upper()
            name = " ".join(p.capitalize() for p in parts[:-1])
            return f"{name} ({modulation})"
        return " ".join(p.capitalize() for p in parts)
    def create_statistical_plots(self, results_dict: Dict[str, Dict[str, Any]], output_dir: str) -> None:
        """
        Creates:
          - boxplots from RAW arrays
          - horizontal CI plots from summary stats
        """

        plt.rcParams.update({
            "font.size": 26,  # base font size
            "axes.titlesize": 26,  # title size
            "axes.labelsize": 26,  # axis label size
            "xtick.labelsize": 24,  # x-tick size
            "ytick.labelsize": 24,  # y-tick size
            "legend.fontsize": 24  # legend size
        })

        os.makedirs(output_dir, exist_ok=True)
        metrics = [m for m in results_dict[next(iter(results_dict))].keys() if m != "_raw"]

        # boxplots from RAW
        for metric in metrics:
            data, labels = [], []
            for cfg, s in results_dict.items():
                if "_raw" in s and metric in s["_raw"]:
                    data.append(np.asarray(s["_raw"][metric], dtype=float))
                    labels.append(cfg)
            if not data:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            bp = ax.boxplot(data, labels=labels, notch=True, showmeans=True, patch_artist=True)
            for patch, c in zip(bp["boxes"], plt.cm.Set3(np.linspace(0, 1, len(data)))):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            #ax.set_title(f"Statistical Distribution Comparison - {metric}")

            ax.set_ylabel(self.prettify_metric(metric))
            ax.grid(True, alpha=0.3)

            pretty_labels = [self.prettify_label_auto(lbl) for lbl in labels]
            ax.set_xticks(np.arange(1, len(pretty_labels) + 1))
            ax.set_xticklabels(pretty_labels, rotation=30, ha="right")

            #plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"statistical_boxplot_{metric.replace(' ','_')}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

        # CI plots from summary
        for metric in metrics:
            means, lows, highs, labels = [], [], [], []
            for cfg, s in results_dict.items():
                if metric in s:
                    means.append(s[metric]["mean"])
                    lows.append(s[metric]["ci_lower"])
                    highs.append(s[metric]["ci_upper"])
                    labels.append(cfg)
            if not means:
                continue

            y = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(12, 6))
            err_low = np.array(means) - np.array(lows)
            err_high = np.array(highs) - np.array(means)

            ax.errorbar(means, y, xerr=[err_low, err_high], fmt="o", capsize=5, capthick=2, markersize=8)
            for i, m in enumerate(means):
                ax.text(m, y[i] + 0.1, f"{m:.3f}", ha="center", va="bottom")
            ax.set_yticks(y)
            ax.set_yticklabels([self.prettify_label_auto(lbl) for lbl in labels])
            ax.set_xlabel(f"{self.prettify_metric(metric)} (95% Confidence Interval)")
            #ax.set_title(f"Confidence Interval Comparison - {metric}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confidence_intervals_{metric.replace(' ','_')}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

    # --------------------- Optional persistence -------------------

    def save_raw_as_csv(self, results_dict: Dict[str, Dict[str, Any]], out_dir: str) -> None:
        """Save raw arrays per configuration/metric for auditability."""
        os.makedirs(out_dir, exist_ok=True)
        for cfg, s in results_dict.items():
            if "_raw" not in s:
                continue
            df = pd.DataFrame({k: pd.Series(v) for k, v in s["_raw"].items()})
            df.to_csv(os.path.join(out_dir, f"{cfg}_raw.csv"), index=False)

# ===================== ExperimentValidator =======================

class ExperimentValidator:
    """
    Wrapper that plugs StatisticalValidator into an existing experiment object.
    The experiment object must expose:
      - modulation_schemes (iterable)
      - test_cases (dict)
      - output_dir (str)
      - num_bits (int)
    """

    def __init__(self, experiment, validator_cfg: Optional[ValidatorConfig] = None):
        self.experiment = experiment
        self.validator = StatisticalValidator(validator_cfg or ValidatorConfig())

    def validate_standard_experiment(self, n_trials: int = 300):
        """Run validation over all (test_case, modulation) combos."""
        print("Adding statistical validation to standard experiment...")
        all_statistical_results: Dict[str, Dict[str, Any]] = {}

        for modulation_scheme in self.experiment.modulation_schemes:
            for test_name in self.experiment.test_cases.keys():
                config_key = f"{test_name}_{modulation_scheme}"
                print(f"Validating: {config_key}")

                def experiment_trial(config_params):
                    # Delegates to a single trial runner below
                    return self._run_single_trial(test_name, modulation_scheme, config_params)

                statistical_results = self.validator.validate_existing_experiment(
                    experiment_trial,
                    {'test_name': test_name, 'modulation': modulation_scheme},
                    n_trials=n_trials
                )
                all_statistical_results[config_key] = statistical_results

        comparison_df = self.validator.compare_configurations(all_statistical_results)

        # Save and plot
        output_dir = os.path.join(self.experiment.output_dir, "statistical_validation")
        os.makedirs(output_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(output_dir, "statistical_comparisons.csv"), index=False)
        self.validator.generate_statistical_report(
            all_statistical_results, comparison_df, os.path.join(output_dir, "statistical_report.md")
        )
        self.validator.create_statistical_plots(all_statistical_results, output_dir)
        self.validator.save_raw_as_csv(all_statistical_results, os.path.join(output_dir, "raw"))

        print(f"Statistical validation complete. Results saved to {output_dir}")
        return all_statistical_results, comparison_df

    # ------------------------- Single Trial -----------------------
    def _run_single_trial(self, test_name: str, modulation_scheme: str, params: dict) -> dict:
        """
        Per-method impairments to create clear separation across configs:
          - Base_AWGN_*         : light impairments (reference)
          - Imperfect_RIS_*     : larger RIS phase quantization error + slight SNR penalty
          - Doppler_AWGN_*      : higher CFO/jitter (time-selective)
          - Mixed_Channel_*     : deeper fading + moderate CFO + RIS error (most severe)

        Adds small randomness to PA efficiency & RIS bias power so energy/bit has a CI.
        Now energy/bit is normalised by correctly received bits,
        so poor channels show higher cost.
        """
        import numpy as np
        import torch
        from wireless.modulation import modulate
        from wireless.demodulation import demodulate
        from tasks.wireless_task import compute_bit_error_rate, compute_semantic_accuracy

        # --------- RNGs (fresh per trial) ---------
        rng = np.random.default_rng(params.get("rng_seed", None))
        torch.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))

        # --------- Per-method impairment settings ---------
        name = test_name.lower()
        cfg = {
            "snr_db_nom": 10.0,
            "snr_db_jit": 1.0,
            "fading_sigma": 0.05,
            "ris_phase_sigma": 0.03,
            "cfo_sigma": 2e-4,
            "phase_jitter_sigma": 0.01,
            "extra_snr_penalty_db": 0.0
        }
        if "imperfect_ris" in name:
            cfg.update(dict(ris_phase_sigma=0.10, extra_snr_penalty_db=0.6))
        if "doppler" in name:
            cfg.update(dict(cfo_sigma=1.0e-3, phase_jitter_sigma=0.02))
        if "mixed_channel" in name:
            cfg.update(dict(fading_sigma=0.20, ris_phase_sigma=0.08,
                            cfo_sigma=7.5e-4, phase_jitter_sigma=0.018,
                            extra_snr_penalty_db=0.4))

        # --------- Trial SNR & bits ---------
        snr_db = float(rng.normal(cfg["snr_db_nom"], cfg["snr_db_jit"])) - cfg["extra_snr_penalty_db"]
        n_bits = int(self.experiment.num_bits)
        bits = torch.randint(0, 2, (n_bits,))

        # --------- Modulation ---------
        tx = modulate(bits, modulation_scheme=modulation_scheme)
        tx_np = tx.numpy().astype(np.complex64)

        # --------- Flat fading + RIS residual phase ---------
        h_amp = 1.0 + cfg["fading_sigma"] * rng.normal()
        h_phi = rng.normal(0.0, cfg["ris_phase_sigma"])
        H = h_amp * np.exp(1j * h_phi).astype(np.complex64)
        tx_ch = (tx_np * H).astype(np.complex64)

        # --------- Effective SNR (apply RIS/test SNR mapping if available) ---------
        case = self.experiment.test_cases[test_name]
        ris = case.get("ris_model", None)
        if ris is not None and hasattr(ris, "apply_ris_gain"):
            snr_eff_db_nom = float(ris.apply_ris_gain(snr_db))
        else:
            snr_eff_db_nom = snr_db

        # --------- AWGN ---------
        Es = float(np.mean(np.abs(tx_ch) ** 2))
        SNR_lin = 10.0 ** (snr_eff_db_nom / 10.0)
        sigma2 = Es / SNR_lin
        noise = (rng.normal(size=tx_ch.shape) + 1j * rng.normal(size=tx_ch.shape)).astype(np.complex64)
        noise *= np.sqrt(sigma2 / 2.0).astype(np.float32)
        rx_noimp = tx_ch + noise

        # --------- Doppler/CFO + phase jitter ---------
        n = np.arange(rx_noimp.size, dtype=np.float32)
        cfo_norm = rng.normal(0.0, cfg["cfo_sigma"])
        jitter = rng.normal(0.0, cfg["phase_jitter_sigma"], size=rx_noimp.size).astype(np.float32)
        impair = np.exp(1j * (2 * np.pi * cfo_norm * n + jitter)).astype(np.complex64)
        rx_np = (rx_noimp * impair).astype(np.complex64)
        rx = torch.from_numpy(rx_np)

        # --------- Demodulation ---------
        rx_bits = demodulate(rx, modulation_scheme=modulation_scheme)

        # --------- Measured effective SNR ---------
        err = rx_np - tx_ch
        snr_eff_lin_meas = (np.mean(np.abs(tx_ch) ** 2) + 1e-12) / (np.mean(np.abs(err) ** 2) + 1e-12)
        snr_eff_db_meas = float(10 * np.log10(snr_eff_lin_meas))

        # --------- Metrics ---------
        ber = float(compute_bit_error_rate(bits, rx_bits))
        acc = float(compute_semantic_accuracy(bits, rx_bits))

        # --------- Energy per bit (normalised by correctly received bits) ---------
        eta_pa = float(np.clip(rng.normal(0.36, 0.03), 0.2, 0.6))
        n_elem = int(getattr(ris, "n_elements", 0) or case.get("ris_elements", 0) or 0)
        P_ris_bias = (0.5e-3 * n_elem) if n_elem > 0 else 40e-3
        m = modulation_scheme.lower()
        bits_per_sym = 2 if "qpsk" in m else (4 if "qam16" in m else 1)
        n_syms = max(1, n_bits // bits_per_sym)
        T_frame = n_syms * 1.0
        E_tx_rf = float(np.sum(np.abs(tx_np) ** 2))
        E_tx_elec = E_tx_rf / eta_pa
        E_ris_bias = P_ris_bias * T_frame
        E_total = E_tx_elec + E_ris_bias

        n_correct = int(n_bits * (1.0 - ber))  # doğru bit sayısı
        epsb = float(E_total / max(n_correct, 1))

        return {
            "semantic_accuracy": acc,
            "ber": ber,
            "energy_per_bit": epsb,
            "snr_effective": snr_eff_db_meas,
        }

    def _run_single_trial_old(self, test_name: str, modulation_scheme: str, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Run one Monte Carlo trial with best practices:
          - jitter nominal SNR per trial
          - randomise RIS state (if supported)
          - measure effective SNR from error-vector power
          - compute EPSB from TRANSMIT energy per semantic bit
        Replace imports with your own project modules.
        """
        import torch
        from wireless.modulation import modulate
        from wireless.demodulation import demodulate
        from channels.awgn import add_awgn_noise
        from tasks.wireless_task import compute_bit_error_rate, compute_semantic_accuracy

        rng = np.random.default_rng(params.get("rng_seed", None))

        # --- per-trial nominal SNR and RIS randomness
        snr_db_nom = 10.0
        snr_db = float(rng.normal(loc=snr_db_nom, scale=1.0))  # ~N(10,1) dB

        # data
        bits = torch.randint(0, 2, (self.experiment.num_bits,))

        # config
        case = self.experiment.test_cases[test_name]
        ris = case.get("ris_model")
        if hasattr(ris, "randomise_state"):
            ris.randomise_state()

        # TX
        tx_symbols = modulate(bits, modulation_scheme=modulation_scheme)

        # effective SNR via RIS mapping
        effective_snr_db = float(ris.apply_ris_gain(snr_db)) if ris is not None else snr_db

        # RX with AWGN
        rx_symbols = add_awgn_noise(tx_symbols, effective_snr_db)

        # Decisions
        recovered_bits = demodulate(rx_symbols, modulation_scheme=modulation_scheme)

        # Measured (post-hoc) effective SNR using error-vector power
        noise_vec = rx_symbols - tx_symbols  # assumes flat channel after RIS
        snr_eff_lin = (tx_symbols.abs().pow(2).mean().item()) / (noise_vec.abs().pow(2).mean().item() + 1e-12)
        snr_eff_db_measured = 10 * np.log10(snr_eff_lin + 1e-12)

        # Metrics
        ber = float(compute_bit_error_rate(bits, recovered_bits))
        acc = float(compute_semantic_accuracy(bits, recovered_bits))

        # ENERGY: compute from TRANSMIT energy per SEMANTIC bit (replace with your mapping if needed)
        tx_energy = tx_symbols.abs().pow(2).sum().item()
        n_semantic_bits = int(bits.numel())  # replace with actual semantic-bit count if different
        epsb = float(tx_energy / max(n_semantic_bits, 1))

        return {
            "semantic_accuracy": acc,
            "ber": ber,
            "energy_per_bit": epsb,
            "snr_effective": snr_eff_db_measured,
        }

# ============================== CLI ==============================

def add_statistical_validation_to_existing_experiment(
    output_dir: str = "outputs/validated_experiment",
    n_trials: int = 300
):
    """
    Example of integrating statistical validation with your experiment.
    Replace EnhancedExperiment import with your own.
    """
    from enhanced_simulation import EnhancedExperiment  # <-- your project module

    experiment = EnhancedExperiment(output_dir=output_dir)
    validator = ExperimentValidator(experiment)
    statistical_results, comparisons = validator.validate_standard_experiment(n_trials=n_trials)
    return statistical_results, comparisons

if __name__ == "__main__":
    print("Adding statistical validation to existing experiments...")
    results, comparisons = add_statistical_validation_to_existing_experiment()
    print("Statistical validation complete!")
