"""
mmd_drift.py
------------
Stage 1 of DriftGuard: label-free concept drift detection using Maximum Mean
Discrepancy (MMD) on autoencoder latent embeddings.

Pipeline:
  1. Load latent embeddings z_train, z_test (from autoencoder.py).
  2. Pick a REFERENCE window from the end of z_train ("normal at train time").
  3. Calibrate a detection THRESHOLD via permutation test on z_train.
     -> threshold = (1 - alpha) quantile of the no-drift null distribution.
  4. Slide a DETECTION window over z_test; at each step compute MMD vs ref.
  5. Flag windows where MMD > threshold as drift alarms.
  6. Plot MMD over time, with the true anomaly fraction per window underneath
     for comparison (the labels are NOT used to set the threshold).

Key idea:
  MMD measures the distance between two distributions in a kernel space.
  MMD ~ 0 means "could be the same distribution"; large MMD means "different".
  By comparing latent embeddings (compressed representations the AE learned),
  we detect when the system's behavior has shifted -- without labels.

Usage:
    python code/mmd_drift.py --tag bgl
    python code/mmd_drift.py --tag hdfs        # control: no drift expected
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="bgl",
                    help="dataset tag (bgl or hdfs)")
parser.add_argument("--ref-size", type=int, default=500,
                    help="number of points in reference window")
parser.add_argument("--win-size", type=int, default=500,
                    help="detection window size")
parser.add_argument("--stride", type=int, default=100,
                    help="step between consecutive detection windows")
parser.add_argument("--n-permute", type=int, default=200,
                    help="permutation rounds for threshold calibration")
parser.add_argument("--alpha", type=float, default=0.01,
                    help="target false-alarm rate (1 - threshold percentile)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_FILE = PROJECT_ROOT / "data" / f"{args.tag}_embeddings.npz"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------------
# Kernel and MMD utilities
# ---------------------------------------------------------------------
def rbf_kernel(X, Y, sigma2):
    """RBF kernel matrix K[i,j] = exp(-||xi - yj||^2 / sigma^2)."""
    XX = (X * X).sum(axis=1, keepdims=True)       # (m, 1)
    YY = (Y * Y).sum(axis=1, keepdims=True)       # (n, 1)
    sqdist = XX + YY.T - 2.0 * X @ Y.T            # (m, n) squared distances
    np.maximum(sqdist, 0, out=sqdist)             # numerical floor
    return np.exp(-sqdist / sigma2)


def median_heuristic_bandwidth(X, max_samples=1000):
    """
    Pick sigma^2 = median pairwise squared distance.
    This is the standard, well-behaved default for the RBF kernel.
    """
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
    """
    Unbiased estimator of squared MMD with RBF kernel:
       MMD^2 = E[K(x,x')] + E[K(y,y')] - 2 E[K(x,y)]
    (excluding self-pairs from the same-distribution terms).
    """
    m = X.shape[0]
    n = Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma2)
    Kyy = rbf_kernel(Y, Y, sigma2)
    Kxy = rbf_kernel(X, Y, sigma2)
    sum_xx = Kxx.sum() - np.trace(Kxx)
    sum_yy = Kyy.sum() - np.trace(Kyy)
    return float(
        sum_xx / (m * (m - 1))
        + sum_yy / (n * (n - 1))
        - 2.0 * Kxy.mean()
    )


def calibrate_threshold(pool, ref_size, win_size, sigma2, n_permute, alpha):
    """
    Build a null distribution by sampling two disjoint random subsets from the
    'pool' (training embeddings) and computing MMD between them. The threshold
    is set at the (1 - alpha) quantile of this null distribution -- so under
    'no drift', we expect alpha fraction of false alarms.
    """
    n = pool.shape[0]
    if n < ref_size + win_size:
        raise ValueError(f"Need >= {ref_size + win_size} points; got {n}")
    null_mmds = np.zeros(n_permute, dtype=np.float64)
    for i in range(n_permute):
        idx = rng.permutation(n)
        A = pool[idx[:ref_size]]
        B = pool[idx[ref_size:ref_size + win_size]]
        null_mmds[i] = mmd2_unbiased(A, B, sigma2)
    threshold = float(np.quantile(null_mmds, 1.0 - alpha))
    return threshold, null_mmds


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"Loading embeddings: {EMB_FILE}")
    data = np.load(EMB_FILE)
    z_train = data["z_train"]
    z_test = data["z_test"]
    y_test = data["y_test"]
    print(f"  z_train: {z_train.shape}, z_test: {z_test.shape}")

    # Reference window: the LAST ref_size training embeddings.
    ref = z_train[-args.ref_size:]
    print(f"  Reference: last {args.ref_size} of z_train")

    # Bandwidth via median heuristic on training embeddings.
    sigma2 = median_heuristic_bandwidth(z_train)
    print(f"  Sigma^2 (median heuristic): {sigma2:.4f}")

    # Calibrate threshold (no labels used).
    print(f"\nCalibrating threshold ({args.n_permute} permutations, "
          f"target alpha={args.alpha})...")
    t0 = time.time()
    threshold, null_mmds = calibrate_threshold(
        z_train, args.ref_size, args.win_size,
        sigma2, args.n_permute, args.alpha,
    )
    print(f"  Null MMD: mean={null_mmds.mean():.4e}, "
          f"max={null_mmds.max():.4e}")
    print(f"  Threshold ({100*(1-args.alpha):.0f}th pct of null): "
          f"{threshold:.4e}")
    print(f"  Calibration took {time.time()-t0:.1f}s.")

    # Slide detection window over test embeddings.
    print(f"\nSliding detection window "
          f"(size={args.win_size}, stride={args.stride}) over z_test...")
    t0 = time.time()
    starts, mmds, anom_rates = [], [], []
    for start in range(0, len(z_test) - args.win_size + 1, args.stride):
        win = z_test[start:start + args.win_size]
        mmds.append(mmd2_unbiased(ref, win, sigma2))
        anom_rates.append(float(y_test[start:start + args.win_size].mean()))
        starts.append(start)
    starts = np.array(starts)
    mmds = np.array(mmds)
    anom_rates = np.array(anom_rates)
    print(f"  Computed {len(mmds)} MMDs in {time.time()-t0:.1f}s.")

    # ------------- Metrics --------------------------------------------------
# Single-window flags (raw, noisy)
    raw_flags = mmds > threshold

    # Sustained-drift criterion: require K consecutive raw alarms.
    # This filters out transient anomaly-induced spikes; keeps real drift.
    K = 3  # consecutive windows required
    drift_flags = np.zeros_like(raw_flags)
    run = 0
    for i, f in enumerate(raw_flags):
        run = run + 1 if f else 0
        if run >= K:
            # Mark this and the previous K-1 windows
            drift_flags[i - K + 1:i + 1] = True

    n_alarms_raw = int(raw_flags.sum())
    n_alarms = int(drift_flags.sum())
    alarm_rate = n_alarms / max(len(drift_flags), 1)

    if drift_flags.any():
        first_idx = int(np.argmax(drift_flags))
        delay = int(starts[first_idx])
    else:
        first_idx, delay = -1, -1

    corr = (float(np.corrcoef(mmds, anom_rates)[0, 1])
            if mmds.std() > 0 and anom_rates.std() > 0 else 0.0)

    print(f"\n--- Drift detection results --------------------------------")
    print(f"  Detection windows:        {len(mmds)}")
    print(f"  Raw alarms (single window > thr): {n_alarms_raw}  "
          f"({n_alarms_raw / max(len(drift_flags), 1):.1%})")
    print(f"  Sustained alarms (K={K} consecutive): {n_alarms}  ({alarm_rate:.1%})")
    if delay >= 0:
        print(f"  First drift at test idx:  {delay}")
    else:
        print(f"  First drift:              never (no drift detected)")
    print(f"  Corr(MMD, true anomaly fraction): {corr:+.3f}")
    print(f"------------------------------------------------------------")

    # ------------- Plot -----------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax = axes[0]
    ax.plot(starts, mmds, label="MMD vs reference", color="steelblue")
    ax.axhline(threshold, color="black", linestyle="--",
               label=f"Threshold ({100*(1-args.alpha):.0f}% null)")
    ax.fill_between(starts, 0, mmds, where=drift_flags,
                    color="crimson", alpha=0.25, label="Drift alarm")
    ax.set_ylabel("MMD²")
    ax.set_title(f"{args.tag.upper()}: label-free drift detection "
                 f"(Stage 1 of DriftGuard)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(starts, anom_rates, color="crimson",
             label="True anomaly fraction in window")
    ax2.set_xlabel("Test index (window start)")
    ax2.set_ylabel("Anomaly frac.")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = RESULTS_DIR / f"{args.tag}_mmd_drift.png"
    fig.savefig(plot_path, dpi=120)
    print(f"\n  Saved plot to {plot_path}")

    # ------------- Summary file --------------------------------------------
    summary_path = RESULTS_DIR / f"{args.tag}_mmd_drift_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"{args.tag.upper()} MMD drift detection "
                f"(Stage 1 of DriftGuard)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Reference size:        {args.ref_size}\n")
        f.write(f"Detection window size: {args.win_size}\n")
        f.write(f"Stride:                {args.stride}\n")
        f.write(f"Bandwidth sigma^2:     {sigma2:.4f}\n")
        f.write(f"Permutations:          {args.n_permute}\n")
        f.write(f"Target false alarm:    {args.alpha}\n")
        f.write(f"Threshold:             {threshold:.4e}\n")
        f.write(f"\nDetection windows: {len(mmds)}\n")
        f.write(f"Alarms:            {n_alarms}  ({alarm_rate:.2%})\n")
        f.write(f"First drift idx:   {delay if delay >= 0 else 'never'}\n")
        f.write(f"Corr(MMD, anomaly frac): {corr:+.3f}\n")
    print(f"  Saved summary to {summary_path}")


if __name__ == "__main__":
    main()