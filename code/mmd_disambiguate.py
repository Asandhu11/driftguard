"""
mmd_disambiguate.py
-------------------
Stage 3 of DriftGuard: distinguish drift from attack on detected events.

For each detection window over the test stream, compute four features:
  1. MMD vs reference (Stage 1 signal: magnitude of deviation)
  2. MMD slope     -- rate of change (sharp spike = attack, gradual = drift)
  3. Template entropy -- Shannon entropy of template usage in the window
                        (high entropy = broad change = drift,
                         low entropy  = narrow change = attack)
  4. (Bonus) True anomaly fraction -- for validation only, not used by rule.

If the slope and entropy axes meaningfully separate "drift-like" from
"attack-like" high-MMD events, we have a working disambiguator.

Inputs:
    data/{tag}_embeddings.npz   (z_train, z_test, y_test)
    data/{tag}_features.npz     (X_test, used for entropy)

Outputs:
    results/{tag}_disambiguate.png   (4-panel diagnostic figure)
    results/{tag}_disambiguate.csv   (per-window features for the report)

Usage:
    python code/mmd_disambiguate.py --tag bgl
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="bgl")
parser.add_argument("--ref-size", type=int, default=500)
parser.add_argument("--win-size", type=int, default=500)
parser.add_argument("--stride", type=int, default=100)
parser.add_argument("--n-permute", type=int, default=200)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--slope-window", type=int, default=5,
                    help="number of windows for the MMD slope estimate")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_FILE = PROJECT_ROOT / "data" / f"{args.tag}_embeddings.npz"
FEAT_FILE = PROJECT_ROOT / "data" / f"{args.tag}_features.npz"

rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------------
# MMD utilities (same as Stage 1, kept self-contained for clarity)
# ---------------------------------------------------------------------
def rbf_kernel(X, Y, sigma2):
    XX = (X * X).sum(axis=1, keepdims=True)
    YY = (Y * Y).sum(axis=1, keepdims=True)
    sqdist = XX + YY.T - 2.0 * X @ Y.T
    np.maximum(sqdist, 0, out=sqdist)
    return np.exp(-sqdist / sigma2)


def median_heuristic_bandwidth(X, max_samples=1000):
    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
    XX = (X * X).sum(axis=1, keepdims=True)
    sqdist = XX + XX.T - 2.0 * X @ X.T
    np.maximum(sqdist, 0, out=sqdist)
    mask = np.triu(np.ones_like(sqdist, dtype=bool), k=1)
    return max(float(np.median(sqdist[mask])), 1e-6)


def mmd2_unbiased(X, Y, sigma2):
    m, n = X.shape[0], Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma2)
    Kyy = rbf_kernel(Y, Y, sigma2)
    Kxy = rbf_kernel(X, Y, sigma2)
    sum_xx = Kxx.sum() - np.trace(Kxx)
    sum_yy = Kyy.sum() - np.trace(Kyy)
    return float(sum_xx / (m * (m - 1))
                 + sum_yy / (n * (n - 1))
                 - 2.0 * Kxy.mean())


def calibrate_threshold(pool, ref_size, win_size, sigma2, n_permute, alpha):
    n = pool.shape[0]
    null_mmds = np.zeros(n_permute, dtype=np.float64)
    for i in range(n_permute):
        idx = rng.permutation(n)
        A = pool[idx[:ref_size]]
        B = pool[idx[ref_size:ref_size + win_size]]
        null_mmds[i] = mmd2_unbiased(A, B, sigma2)
    return float(np.quantile(null_mmds, 1.0 - alpha))


# ---------------------------------------------------------------------
# Stage 3 feature: template entropy per detection window
# ---------------------------------------------------------------------
def template_entropy_per_window(X_test, starts, win_size):
    """
    Shannon entropy of the aggregated template-count distribution
    within each detection window.

    High entropy  -> templates used broadly (system-wide change, drift-like)
    Low  entropy  -> templates concentrated on a few (narrow event, attack-like)
    """
    entropies = np.zeros(len(starts))
    for i, s in enumerate(starts):
        counts = X_test[s:s + win_size].sum(axis=0)
        total = counts.sum()
        if total <= 0:
            entropies[i] = 0.0
            continue
        p = counts / total
        p = p[p > 0]  # 0 * log 0 = 0; skip zero entries
        entropies[i] = -float(np.sum(p * np.log(p)))
    return entropies


def main():
    # ---- Load -----------------------------------------------------------
    print(f"Loading embeddings: {EMB_FILE}")
    data = np.load(EMB_FILE)
    z_train = data["z_train"]
    z_test = data["z_test"]
    y_test = data["y_test"]
    print(f"  z_train: {z_train.shape}, z_test: {z_test.shape}")

    print(f"Loading features:  {FEAT_FILE}")
    feat = np.load(FEAT_FILE)
    X_test = feat["X_test"]
    print(f"  X_test:  {X_test.shape}")

    # ---- Re-compute MMD curve (same as Stage 1) ------------------------
    ref = z_train[-args.ref_size:]
    sigma2 = median_heuristic_bandwidth(z_train)
    threshold = calibrate_threshold(
        z_train, args.ref_size, args.win_size,
        sigma2, args.n_permute, args.alpha,
    )
    print(f"\nThreshold: {threshold:.4e}  (sigma^2={sigma2:.4f})")

    print("Computing MMD curve...")
    t0 = time.time()
    starts, mmds, anom_rates = [], [], []
    for s in range(0, len(z_test) - args.win_size + 1, args.stride):
        win = z_test[s:s + args.win_size]
        mmds.append(mmd2_unbiased(ref, win, sigma2))
        anom_rates.append(float(y_test[s:s + args.win_size].mean()))
        starts.append(s)
    starts = np.array(starts)
    mmds = np.array(mmds)
    anom_rates = np.array(anom_rates)
    print(f"  {len(mmds)} windows in {time.time() - t0:.1f}s")

    # ---- Feature 1: MMD slope over the last K windows -------------------
    K = args.slope_window
    slopes = np.zeros(len(mmds))
    for i in range(len(mmds)):
        lo = max(0, i - K + 1)
        x = np.arange(lo, i + 1)
        y = mmds[lo:i + 1]
        if len(x) >= 2:
            slopes[i] = float(np.polyfit(x, y, 1)[0])

    # ---- Feature 2: template entropy per window ------------------------
    print("Computing template entropies...")
    entropies = template_entropy_per_window(X_test, starts, args.win_size)

    # ---- Identify high-MMD events --------------------------------------
    high_mmd = mmds > threshold
    high_idx = np.where(high_mmd)[0]
    print(f"\nHigh-MMD windows: {len(high_idx)} of {len(mmds)}")

    if len(high_idx) > 5:
        ha = anom_rates[high_idx]
        hs = slopes[high_idx]
        he = entropies[high_idx]
        print(f"  Anomaly fraction: mean={ha.mean():.3f}, max={ha.max():.3f}")
        print(f"  Slope:            mean={hs.mean():.4f}, "
              f"min={hs.min():.4f}, max={hs.max():.4f}")
        print(f"  Entropy:          mean={he.mean():.3f}, "
              f"min={he.min():.3f}, max={he.max():.3f}")
        corr_slope = float(np.corrcoef(hs, ha)[0, 1])
        corr_ent = float(np.corrcoef(he, ha)[0, 1])
        print(f"\n  Corr(slope,   anomaly fraction): {corr_slope:+.3f}")
        print(f"  Corr(entropy, anomaly fraction): {corr_ent:+.3f}")
        # Positive corr(slope, anomaly) -> attacks coincide with steep MMD rises
        # Negative corr(entropy, anomaly) -> attacks coincide with narrow template use

    # ---- 4-panel diagnostic plot ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(starts, mmds, color="steelblue", label="MMD")
    ax.axhline(threshold, color="black", linestyle="--", label="Threshold")
    ax.fill_between(starts, 0, mmds, where=high_mmd,
                    color="crimson", alpha=0.2, label="High-MMD")
    ax.set_xlabel("Test idx"); ax.set_ylabel("MMD²")
    ax.set_title("MMD curve"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(starts, anom_rates, color="crimson")
    ax.set_xlabel("Test idx"); ax.set_ylabel("Anomaly fraction")
    ax.set_title("True anomaly fraction (reference)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(starts, slopes, color="purple")
    ax.axhline(0, color="black", linestyle=":")
    ax.set_xlabel("Test idx"); ax.set_ylabel("MMD slope")
    ax.set_title(f"MMD slope (window K={K})"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(high_idx) > 0:
        sc = ax.scatter(slopes[high_idx], entropies[high_idx],
                        c=anom_rates[high_idx], cmap="coolwarm",
                        s=60, alpha=0.85,
                        edgecolors="black", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="True anomaly fraction")
    ax.axvline(0, color="black", linestyle=":")
    ax.set_xlabel("MMD slope  (negative=stable, positive=rising)")
    ax.set_ylabel("Template entropy  (higher = broader change)")
    ax.set_title("High-MMD events: drift (top-left) vs attack (bottom-right)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{args.tag.upper()} Stage 3: drift vs attack disambiguation",
                 fontsize=14)
    fig.tight_layout()
    plot_path = RESULTS_DIR / f"{args.tag}_disambiguate.png"
    fig.savefig(plot_path, dpi=120)
    print(f"\nSaved plot to {plot_path}")

    # ---- Per-window CSV (for the report) -------------------------------
    csv_path = RESULTS_DIR / f"{args.tag}_disambiguate.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "mmd", "high_mmd", "slope",
                    "entropy", "anomaly_fraction"])
        for i in range(len(starts)):
            w.writerow([int(starts[i]), float(mmds[i]),
                        int(high_mmd[i]), float(slopes[i]),
                        float(entropies[i]), float(anom_rates[i])])
    print(f"Saved per-window features to {csv_path}")


if __name__ == "__main__":
    main()