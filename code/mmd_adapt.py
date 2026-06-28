"""
mmd_adapt.py
------------
Stage 2 of DriftGuard: selective replay-based adaptation.

When Stage 1 detects drift in a region of the stream:
  1. Identify the drift region (from Stage 1 plot).
  2. Split it: first part for adaptation, rest held out for evaluation.
  3. Within the adaptation portion, score every window with the CURRENT
     model and keep the bottom k% by reconstruction error: these are
     "drifted normal" candidates the model already kind-of recognizes.
  4. Build a REPLAY BUFFER from the original training set so we don't
     forget the old normal (anti-catastrophic-forgetting).
  5. Mix candidates + replay; fine-tune the AE for a few epochs at a small LR.
  6. Evaluate before vs after on three slices: pre-drift, held-out drift,
     and full test.

Inputs:
    results/{tag}_autoencoder.pt
    data/{tag}_features.npz

Outputs:
    results/{tag}_autoencoder_adapted.pt
    results/{tag}_adapt_results.txt
    results/{tag}_adapt_comparison.png

Usage:
    python code/mmd_adapt.py --tag bgl --drift-start-idx 3500
"""

import argparse
import copy
import time
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
parser.add_argument("--drift-start-idx", type=int, default=3500,
                    help="test index where sustained drift begins "
                         "(read off Stage 1 plot)")
parser.add_argument("--candidate-frac", type=float, default=0.5,
                    help="fraction of adapt-region samples to keep as "
                         "'drifted normal' (lowest reconstruction error)")
parser.add_argument("--adapt-frac", type=float, default=0.4,
                    help="fraction of drift region used for adaptation "
                         "(rest held out for evaluation)")
parser.add_argument("--replay-mult", type=float, default=1.0,
                    help="replay buffer size = candidates * this multiplier")
parser.add_argument("--adapt-epochs", type=int, default=10)
parser.add_argument("--adapt-batch", type=int, default=16)
parser.add_argument("--adapt-lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_FILE = PROJECT_ROOT / "data" / f"{args.tag}_features.npz"
MODEL_FILE = RESULTS_DIR / f"{args.tag}_autoencoder.pt"
TAG = args.tag

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------------
# Model definition (matches autoencoder.py)
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def compute_errors(model, X):
    """Per-sample reconstruction MSE on log1p-normalized X."""
    X_norm = torch.from_numpy(np.log1p(X).astype(np.float32))
    model.eval()
    with torch.no_grad():
        x_hat = model(X_norm)
        return ((x_hat - X_norm) ** 2).mean(dim=1).numpy()


def best_f1(y, scores):
    """Best F1 across all thresholds + AUC. Returns dict."""
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "threshold": float("nan")}
    precision, recall, thresh = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    i = int(np.argmax(f1))
    return {
        "auc": float(roc_auc_score(y, scores)),
        "f1": float(f1[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "threshold": float(thresh[min(i, len(thresh) - 1)]),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"Tag: {TAG}")
    print(f"Drift start (from Stage 1): test idx {args.drift_start_idx}")

    # ---- Load features and trained model -------------------------------
    data = np.load(FEATURES_FILE)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    ckpt = torch.load(MODEL_FILE, weights_only=False)
    input_dim = ckpt["input_dim"]
    model = Autoencoder(input_dim)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded {MODEL_FILE}")

    # Keep frozen copy for "before" evaluation
    pre_model = copy.deepcopy(model)

    # ---- Define pre-drift vs drift portions of the test set -----------
    pre_mask = np.arange(len(X_test)) < args.drift_start_idx
    drift_idx = np.where(~pre_mask)[0]
    print(f"Pre-drift portion: {pre_mask.sum():,} samples")
    print(f"Drift portion:     {len(drift_idx):,} samples")

    # ---- Split drift region: adaptation prefix + held-out eval ----------
    n_adapt = int(args.adapt_frac * len(drift_idx))
    adapt_idx = drift_idx[:n_adapt]
    eval_drift_idx = drift_idx[n_adapt:]
    print(f"Adapt prefix:      {len(adapt_idx):,}")
    print(f"Held-out drift:    {len(eval_drift_idx):,}")

    # ---- Score adaptation candidates with the CURRENT model -----------
    X_cand = X_test[adapt_idx]
    err_cand = compute_errors(model, X_cand)

    n_keep = int(args.candidate_frac * len(err_cand))
    order = np.argsort(err_cand)
    drifted_normal_idx = adapt_idx[order[:n_keep]]
    X_drifted_normal = X_test[drifted_normal_idx]
    print(f"Drifted-normal kept: {len(X_drifted_normal):,} "
          f"(bottom {100 * args.candidate_frac:.0f}% by recon error)")

    # ---- Replay buffer from original training set ---------------------
    n_replay = min(int(args.replay_mult * len(X_drifted_normal)), len(X_train))
    replay_idx = rng.choice(len(X_train), n_replay, replace=False)
    X_replay = X_train[replay_idx]
    print(f"Replay buffer:       {len(X_replay):,}")

    X_mix = np.vstack([X_drifted_normal, X_replay])
    X_mix_t = torch.from_numpy(np.log1p(X_mix).astype(np.float32))
    print(f"Mixed adaptation set: {len(X_mix):,}")

    # ---- Fine-tune ----------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.adapt_lr)
    criterion = nn.MSELoss()

    print(f"\nFine-tuning: epochs={args.adapt_epochs}, "
          f"batch={args.adapt_batch}, lr={args.adapt_lr}")
    t0 = time.time()
    model.train()
    for epoch in range(1, args.adapt_epochs + 1):
        perm = torch.randperm(len(X_mix_t))
        running, n_batches = 0.0, 0
        for s in range(0, len(X_mix_t), args.adapt_batch):
            batch = X_mix_t[perm[s:s + args.adapt_batch]]
            optimizer.zero_grad()
            xh = model(batch)
            loss = criterion(xh, batch)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        print(f"  Epoch {epoch:>2}  loss = {running / n_batches:.6f}")
    print(f"Adapted in {time.time() - t0:.1f}s.")

    # ---- Evaluate before vs after on three slices ----------------------
    def eval_on(mdl, X, y):
        e = compute_errors(mdl, X)
        return best_f1(y, e)

    Xp, yp = X_test[pre_mask], y_test[pre_mask]
    Xd, yd = X_test[eval_drift_idx], y_test[eval_drift_idx]
    Xf, yf = X_test, y_test

    print("\n" + "=" * 80)
    print(f"{'Slice':<22}{'Anom rate':>11}"
          f"{'AUC before':>13}{'AUC after':>12}"
          f"{'F1 before':>12}{'F1 after':>11}")
    print("-" * 80)

    results = {}
    for name, X, y in [("Pre-drift", Xp, yp),
                       ("Drift (held out)", Xd, yd),
                       ("Full test", Xf, yf)]:
        b = eval_on(pre_model, X, y)
        a = eval_on(model, X, y)
        ar = float(y.mean())
        print(f"{name:<22}{ar:>11.2%}"
              f"{b['auc']:>13.4f}{a['auc']:>12.4f}"
              f"{b['f1']:>12.4f}{a['f1']:>11.4f}")
        results[name] = {"before": b, "after": a, "anom_rate": ar}
    print("=" * 80)

    # ---- Save artifacts ------------------------------------------------
    out_model = RESULTS_DIR / f"{TAG}_autoencoder_adapted.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "tag": TAG,
        "adapted": True,
        "drift_start_idx": args.drift_start_idx,
    }, out_model)
    print(f"\nSaved adapted model to {out_model}")

    summary_path = RESULTS_DIR / f"{TAG}_adapt_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"{TAG.upper()} DriftGuard Stage 2: selective replay adaptation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Drift start idx:       {args.drift_start_idx}\n")
        f.write(f"Adapt prefix size:     {len(adapt_idx)}\n")
        f.write(f"Drifted-normal kept:   {len(X_drifted_normal)} "
                f"(bottom {100 * args.candidate_frac:.0f}% by recon error)\n")
        f.write(f"Replay buffer:         {len(X_replay)}\n")
        f.write(f"Epochs / batch / lr:   {args.adapt_epochs} / "
                f"{args.adapt_batch} / {args.adapt_lr}\n")
        for name, r in results.items():
            f.write(f"\n{name}  (anomaly rate {r['anom_rate']:.2%})\n")
            f.write(f"  AUC: {r['before']['auc']:.4f}  ->  {r['after']['auc']:.4f}\n")
            f.write(f"  F1:  {r['before']['f1']:.4f}  ->  {r['after']['f1']:.4f}\n")
            f.write(f"  P:   {r['before']['precision']:.4f}  ->  {r['after']['precision']:.4f}\n")
            f.write(f"  R:   {r['before']['recall']:.4f}  ->  {r['after']['recall']:.4f}\n")
    print(f"Saved summary to {summary_path}")

    # ---- Plot before vs after ------------------------------------------
    slices = list(results.keys())
    f1_b = [results[s]["before"]["f1"] for s in slices]
    f1_a = [results[s]["after"]["f1"] for s in slices]
    auc_b = [results[s]["before"]["auc"] for s in slices]
    auc_a = [results[s]["after"]["auc"] for s in slices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(slices))
    w = 0.35
    ax1.bar(x - w / 2, f1_b, w, label="Before", color="steelblue")
    ax1.bar(x + w / 2, f1_a, w, label="After", color="crimson")
    ax1.set_xticks(x); ax1.set_xticklabels(slices, rotation=15, ha="right")
    ax1.set_ylabel("F1"); ax1.set_title(f"{TAG.upper()}: F1 before vs after")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.bar(x - w / 2, auc_b, w, label="Before", color="steelblue")
    ax2.bar(x + w / 2, auc_a, w, label="After", color="crimson")
    ax2.set_xticks(x); ax2.set_xticklabels(slices, rotation=15, ha="right")
    ax2.set_ylabel("ROC-AUC"); ax2.set_title(f"{TAG.upper()}: AUC before vs after")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_plot = RESULTS_DIR / f"{TAG}_adapt_comparison.png"
    fig.savefig(out_plot, dpi=120)
    print(f"Saved plot to {out_plot}")


if __name__ == "__main__":
    main()