"""
stage2_candidate_sweep.py
-------------------------
Sweeps candidate_frac (fraction of adapt-prefix windows treated as
'drifted normal') for Stage 2. Replay multiplier and epochs are fixed.

Hypothesis: the dominant Stage-2 lever on BGL is candidate selection,
not replay buffer size. Low candidate_frac is conservative (few samples,
little adaptation gain). High candidate_frac increases the chance of
contaminating training with actual anomalies, which can hurt drift gain.

Each run starts from the same frozen pre-trained autoencoder.

Outputs:
    results/{tag}_stage2_candsweep.csv
    results/{tag}_stage2_candsweep.png

Usage:
    python code/stage2_candidate_sweep.py --tag bgl --drift-start-idx 3500
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="bgl")
parser.add_argument("--drift-start-idx", type=int, default=3500)
parser.add_argument("--adapt-frac", type=float, default=0.4)
parser.add_argument("--replay-mult", type=float, default=1.0)
parser.add_argument("--adapt-epochs", type=int, default=10)
parser.add_argument("--adapt-batch", type=int, default=16)
parser.add_argument("--adapt-lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Candidate fractions to sweep over.
CAND_FRACS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_FILE = PROJECT_ROOT / "data" / f"{args.tag}_features.npz"
MODEL_FILE = RESULTS_DIR / f"{args.tag}_autoencoder.pt"
TAG = args.tag


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def get_layer_sizes(input_dim):
    if input_dim < 100:   return 32, 16, 8
    elif input_dim < 1000: return 128, 32, 16
    else:                  return 256, 64, 16


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h1, h2, lat = get_layer_sizes(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),         nn.ReLU(),
            nn.Linear(h2, lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(lat, h2), nn.ReLU(),
            nn.Linear(h2, h1),  nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def compute_errors(model, X):
    X_norm = torch.from_numpy(np.log1p(X).astype(np.float32))
    model.eval()
    with torch.no_grad():
        return ((model(X_norm) - X_norm) ** 2).mean(dim=1).numpy()


def best_metrics(y, scores):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return auc, float(np.max(f1))


def adapt(model, X_drifted_normal, X_replay, lr, epochs, batch):
    if len(X_replay) > 0:
        X_mix = np.vstack([X_drifted_normal, X_replay])
    else:
        X_mix = X_drifted_normal
    X_mix_t = torch.from_numpy(np.log1p(X_mix).astype(np.float32))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_mix_t))
        for s in range(0, len(X_mix_t), batch):
            xb = X_mix_t[perm[s:s + batch]]
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"Tag: {TAG}")
    print(f"Drift start idx: {args.drift_start_idx}")
    print(f"Fixed: replay_mult={args.replay_mult}, "
          f"epochs={args.adapt_epochs}, lr={args.adapt_lr}")

    data = np.load(FEATURES_FILE)
    X_train, X_test, y_test = data["X_train"], data["X_test"], data["y_test"]
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    ckpt = torch.load(MODEL_FILE, weights_only=False)
    input_dim = ckpt["input_dim"]
    base_state = ckpt["state_dict"]
    print(f"Loaded base model from {MODEL_FILE}")

    pre_mask = np.arange(len(X_test)) < args.drift_start_idx
    drift_idx = np.where(~pre_mask)[0]
    n_adapt = int(args.adapt_frac * len(drift_idx))
    adapt_idx = drift_idx[:n_adapt]
    eval_drift_idx = drift_idx[n_adapt:]
    print(f"Pre-drift: {pre_mask.sum()},  "
          f"adapt prefix: {n_adapt},  held-out: {len(eval_drift_idx)}")

    # Score candidates ONCE with the base model -- ordering is fixed.
    base_model = Autoencoder(input_dim)
    base_model.load_state_dict(base_state)
    err_cand = compute_errors(base_model, X_test[adapt_idx])
    order = np.argsort(err_cand)

    # Ground-truth anomaly fraction by candidate-frac level (diagnostic).
    y_adapt = y_test[adapt_idx]

    Xp, yp = X_test[pre_mask], y_test[pre_mask]
    Xd, yd = X_test[eval_drift_idx], y_test[eval_drift_idx]
    Xf, yf = X_test, y_test

    pre_model = Autoencoder(input_dim)
    pre_model.load_state_dict(base_state)
    auc_pre_b, f1_pre_b = best_metrics(yp, compute_errors(pre_model, Xp))
    auc_drift_b, f1_drift_b = best_metrics(yd, compute_errors(pre_model, Xd))
    auc_full_b, f1_full_b = best_metrics(yf, compute_errors(pre_model, Xf))

    rng = np.random.default_rng(args.seed)
    results = []
    for cf in CAND_FRACS:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        n_keep = max(int(cf * len(err_cand)), 1)
        drifted_normal_idx = adapt_idx[order[:n_keep]]
        X_drifted_normal = X_test[drifted_normal_idx]
        # How many of these are actually labeled anomalous?
        contamination = float(y_test[drifted_normal_idx].mean())

        n_replay = min(int(args.replay_mult * n_keep), len(X_train))
        if n_replay > 0:
            replay_idx = rng.choice(len(X_train), n_replay, replace=False)
            X_replay = X_train[replay_idx]
        else:
            X_replay = np.zeros((0, X_train.shape[1]), dtype=X_train.dtype)

        model = Autoencoder(input_dim)
        model.load_state_dict(base_state)
        adapt(model, X_drifted_normal, X_replay,
              args.adapt_lr, args.adapt_epochs, args.adapt_batch)

        auc_pre,   f1_pre   = best_metrics(yp, compute_errors(model, Xp))
        auc_drift, f1_drift = best_metrics(yd, compute_errors(model, Xd))
        auc_full,  f1_full  = best_metrics(yf, compute_errors(model, Xf))

        results.append({
            "cand_frac": cf, "n_keep": n_keep,
            "contamination": contamination,
            "auc_pre": auc_pre,     "f1_pre": f1_pre,
            "auc_drift": auc_drift, "f1_drift": f1_drift,
            "auc_full": auc_full,   "f1_full": f1_full,
        })
        print(f"cand_frac={cf:>4.1f}  n={n_keep:>4d}  "
              f"contam={contamination:.1%}   "
              f"pre F1={f1_pre:.3f}  drift F1={f1_drift:.3f}  "
              f"drift AUC={auc_drift:.3f}")

    # ---- Save CSV ------------------------------------------------------
    csv_path = RESULTS_DIR / f"{TAG}_stage2_candsweep.csv"
    with open(csv_path, "w") as f:
        f.write("cand_frac,n_keep,contamination,"
                "auc_pre,f1_pre,auc_drift,f1_drift,auc_full,f1_full\n")
        f.write(f"baseline,0,0,"
                f"{auc_pre_b:.4f},{f1_pre_b:.4f},"
                f"{auc_drift_b:.4f},{f1_drift_b:.4f},"
                f"{auc_full_b:.4f},{f1_full_b:.4f}\n")
        for r in results:
            f.write(f"{r['cand_frac']:.2f},{r['n_keep']},"
                    f"{r['contamination']:.4f},"
                    f"{r['auc_pre']:.4f},{r['f1_pre']:.4f},"
                    f"{r['auc_drift']:.4f},{r['f1_drift']:.4f},"
                    f"{r['auc_full']:.4f},{r['f1_full']:.4f}\n")
    print(f"\nSaved CSV to {csv_path}")

    # ---- Plot ----------------------------------------------------------
    cfs = [r["cand_frac"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # F1 panel
    ax1.plot(cfs, [r["f1_pre"]   for r in results], "o-",
             color="steelblue", label="Pre-drift")
    ax1.plot(cfs, [r["f1_drift"] for r in results], "s-",
             color="crimson",   label="Drift (held out)")
    ax1.axhline(f1_pre_b,   color="steelblue", linestyle=":", alpha=0.6,
                label="Pre-drift baseline")
    ax1.axhline(f1_drift_b, color="crimson",   linestyle=":", alpha=0.6,
                label="Drift baseline")
    ax1.set_xlabel("Candidate fraction")
    ax1.set_ylabel("Best F1")
    ax1.set_title(f"{TAG.upper()} Stage 2: F1 vs candidate fraction")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # AUC + contamination panel
    ax2.plot(cfs, [r["auc_drift"] for r in results], "s-",
             color="crimson", label="Drift AUC")
    ax2.axhline(auc_drift_b, color="crimson", linestyle=":", alpha=0.6,
                label="Drift baseline")
    ax2.set_xlabel("Candidate fraction")
    ax2.set_ylabel("ROC-AUC", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(cfs, [r["contamination"] for r in results], "d--",
              color="black", label="Anomaly contamination")
    ax2b.set_ylabel("Anomaly contamination", color="black")
    ax2b.tick_params(axis="y", labelcolor="black")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right")
    ax2.set_title(f"{TAG.upper()} Stage 2: drift AUC + contamination")

    fig.suptitle(f"{TAG.upper()} Stage 2 candidate-fraction sweep "
                 f"(replay={args.replay_mult}, epochs={args.adapt_epochs})",
                 fontsize=13)
    fig.tight_layout()
    plot_path = RESULTS_DIR / f"{TAG}_stage2_candsweep.png"
    fig.savefig(plot_path, dpi=120)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()