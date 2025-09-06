import matplotlib
matplotlib.use("Agg")  # Headless backend for script usage

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – ensures 3‑D proj registered

# -----------------------------------------------------------------------------
# Global styling – tuned for academic‑journal figures
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "axes.grid": True,
    "axes.grid.which": "both",
})

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _poly_regression(x: np.ndarray,
                     y: np.ndarray,
                     deg: int = 3,
                     num: int = 400):
    coeffs = np.polyfit(x, y, deg)
    x_smooth = np.linspace(x.min(), x.max(), num)
    y_smooth = np.polyval(coeffs, x_smooth)
    return x_smooth, y_smooth

# -----------------------------------------------------------------------------
# Public plotting API
# -----------------------------------------------------------------------------

def _style_snr_axis(ax, snr_range):
    ax.set_xlim([min(snr_range), max(snr_range)])
    if min(snr_range) < -10:
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.text(0, ax.get_ylim()[1]*0.95, '0 dB', ha='left', va='top', fontsize=8, color='gray')


def plot_accuracy_vs_snr(snr_list, accuracy_list, title="Accuracy vs SNR", save_path=None, regression_deg=3):
    snr = np.asarray(snr_list, dtype=float)
    acc = np.asarray(accuracy_list, dtype=float)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.plot(snr, acc, linestyle="--", linewidth=0.9, alpha=0.7)
    if len(snr) > regression_deg:
        x_s, y_s = _poly_regression(snr, acc, deg=regression_deg)
        ax.plot(x_s, y_s, linewidth=1.4)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Semantic Accuracy")
    ax.set_title(title)
    _style_snr_axis(ax, snr)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_energy_vs_snr(snr_list, epsb_list, title="Energy per Semantic Bit vs SNR", save_path=None, regression_deg=3):
    snr = np.asarray(snr_list, dtype=float)
    epsb = np.asarray(epsb_list, dtype=float)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.plot(snr, epsb, linestyle="--", linewidth=0.9, alpha=0.7)
    if len(snr) > regression_deg:
        x_s, y_s = _poly_regression(snr, epsb, deg=regression_deg)
        ax.plot(x_s, y_s, linewidth=1.4)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Energy per Semantic Bit (normalised)")
    ax.set_title(title)
    _style_snr_axis(ax, snr)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_ber_vs_snr(snr_list, ber_list, title="BER vs SNR", save_path=None, regression_deg=3):
    snr = np.asarray(snr_list, dtype=float)
    ber = np.asarray(ber_list, dtype=float)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.plot(snr, ber, linestyle="--", linewidth=0.9, alpha=0.7)
    if len(snr) > regression_deg:
        x_s, y_s = _poly_regression(snr, ber, deg=regression_deg)
        ax.plot(x_s, y_s, linewidth=1.4)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(title)
    _style_snr_axis(ax, snr)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_ris_elements(elements_list, accuracy_list, title="Semantic Accuracy vs RIS Elements", save_path=None, regression_deg=2):
    elements = np.asarray(elements_list, dtype=float)
    acc = np.asarray(accuracy_list, dtype=float)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.plot(elements, acc, linestyle="--", linewidth=0.9, alpha=0.7)
    if len(elements) > regression_deg:
        x_s, y_s = _poly_regression(elements, acc, deg=regression_deg)
        ax.plot(x_s, y_s, linewidth=1.4)

    ax.set_xlabel("RIS Elements")
    ax.set_ylabel("Average Semantic Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim([min(elements), max(elements)])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_constellation(symbols, title="Constellation Diagram", save_path=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(symbols.real, symbols.imag, s=8, alpha=0.6, edgecolors="none")

    ax.set_xlabel("In‑phase")
    ax.set_ylabel("Quadrature")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect("equal", "box")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_rdr_surface(bit_rates, distortions, relevances, title="Rate‑Distortion‑Relevance Surface", save_path=None):
    br = np.asarray(bit_rates, dtype=float)
    d = np.asarray(distortions, dtype=float)
    r = np.asarray(relevances, dtype=float)

    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(111, projection="3d")

    rng = np.random.default_rng(seed=0)
    br_j = br + rng.normal(0, 0.05, size=br.shape)
    d_j = d + rng.normal(0, 0.005, size=d.shape)

    trisurf = ax.plot_trisurf(br_j, d_j, r, cmap="viridis", edgecolor="none", alpha=0.88)

    if len(br) >= 3:
        A = np.c_[br, d, np.ones_like(br)]
        a, b, c = np.linalg.lstsq(A, r, rcond=None)[0]
        br_lin = np.linspace(br.min(), br.max(), 24)
        d_lin = np.linspace(d.min(), d.max(), 24)
        BR, D = np.meshgrid(br_lin, d_lin)
        R = a * BR + b * D + c
        ax.plot_surface(BR, D, R, alpha=0.25, color="crimson", linewidth=0, antialiased=False)

    ax.set_xlabel("Bit Rate")
    ax.set_ylabel("Distortion (BER)")
    ax.set_zlabel("Semantic Relevance (Accuracy)")
    ax.set_title(title)
    fig.colorbar(trisurf, shrink=0.5, aspect=12, pad=0.08, label="Relevance")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
