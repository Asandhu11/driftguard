"""
autoencoder.py
--------------
Trains a feed-forward autoencoder for log anomaly detection.

Works on any features.npz file with the format produced by build_features.py
(HDFS) or build_bgl_features.py (BGL). The hidden layer sizes scale with
input dimension so the same script handles HDFS (47 dims) and BGL (1822 dims).

Usage from the project root:
    python code/autoencoder.py                                    # default: HDFS
    python code/autoencoder.py --features bgl_features.npz --tag bgl

Outputs (named by --tag):
    results/{tag}_recon_error_histogram.png
    results/{tag}_training_loss.png
    results/{tag}_autoencoder_results.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--features", type=str, default="features.npz",
                    help="Path to features .npz file (relative to data/)")
parser.add_argument("--tag", type=str, default="hdfs",
                    help="Tag used in output filenames (e.g. 'hdfs', 'bgl')")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = PROJECT_ROOT / "data" / args.features
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TAG = args.tag

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_layer_sizes(input_dim):
    """Pick reasonable hidden + latent sizes given input dim."""
    if input_dim < 100:
        return 32, 16, 8      # small (HDFS scale)
    elif input_dim < 1000:
        return 128, 32, 16
    else:
        return 256, 64, 16    # large (BGL scale)


class Autoencoder(nn.Module):
    """Symmetric encoder + decoder. Architecture scales with input_dim."""

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
        z = self.encoder(x)
        return self.decoder(z)


def main():
    # -----------------------------------------------------------------
    # 1. Load features
    # -----------------------------------------------------------------
    print(f"Loading features from {FEATURES_FILE} ...")
    data = np.load(FEATURES_FILE)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Test anomaly rate: {y_test.mean():.2%}")

    # -----------------------------------------------------------------
    # 2. Normalize with log1p (counts can vary widely)
    # -----------------------------------------------------------------
    X_train_norm = np.log1p(X_train).astype(np.float32)
    X_test_norm = np.log1p(X_test).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_norm)
    X_test_t = torch.from_numpy(X_test_norm)

    train_loader = DataLoader(
        TensorDataset(X_train_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # -----------------------------------------------------------------
    # 3. Build model
    # -----------------------------------------------------------------
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    h1, h2, lat = get_layer_sizes(input_dim)
    print(f"\nArchitecture: {input_dim} -> {h1} -> {h2} -> {lat} -> "
          f"{h2} -> {h1} -> {input_dim}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # -----------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (batch size {args.batch_size})...")
    train_losses = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n_batches = 0.0, 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            x_hat = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            n_batches += 1
        avg = run_loss / max(n_batches, 1)
        train_losses.append(avg)
        print(f"  Epoch {epoch:>2}/{args.epochs}   loss = {avg:.6f}")
    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed:.1f}s.")

    # -----------------------------------------------------------------
    # 5. Evaluate: reconstruction error per test sample
    # -----------------------------------------------------------------
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        x_hat = model(X_test_t)
        errors = ((x_hat - X_test_t) ** 2).mean(dim=1).numpy()

    auc = roc_auc_score(y_test, errors)
    precision, recall, thresholds = precision_recall_curve(y_test, errors)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1))
    best_f1 = f1[best_idx]
    best_p = precision[best_idx]
    best_r = recall[best_idx]
    best_thr = thresholds[min(best_idx, len(thresholds) - 1)]

    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Best F1:   {best_f1:.4f}  (P={best_p:.4f}, R={best_r:.4f})")
    print(f"  Threshold: {best_thr:.6f}")

    # -----------------------------------------------------------------
    # 5b. Save the trained model and its latent embeddings.
    # We need these in Week 3 to compute MMD drift scores.
    # -----------------------------------------------------------------
    model_path = RESULTS_DIR / f"{TAG}_autoencoder.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "tag": TAG,
    }, model_path)
    print(f"  Saved trained model to {model_path}")

    # Extract latent embeddings for training and test sets.
    print("Extracting latent embeddings...")
    with torch.no_grad():
        z_train = model.encoder(X_train_t).numpy().astype(np.float32)
        z_test = model.encoder(X_test_t).numpy().astype(np.float32)
    embeddings_path = PROJECT_ROOT / "data" / f"{TAG}_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        z_train=z_train,
        z_test=z_test,
        y_test=y_test,
    )
    print(f"  Saved embeddings to {embeddings_path}")
    print(f"  z_train shape: {z_train.shape}, z_test shape: {z_test.shape}")

    # -----------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    upper = float(np.percentile(errors, 99))
    bins = np.linspace(0, upper, 80)
    ax.hist(errors[y_test == 0], bins=bins, alpha=0.6,
            label="Normal", color="steelblue", density=True)
    ax.hist(errors[y_test == 1], bins=bins, alpha=0.6,
            label="Anomalous", color="crimson", density=True)
    ax.axvline(best_thr, color="black", linestyle="--",
               label=f"Best-F1 threshold = {best_thr:.4f}")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title(f"{TAG.upper()} autoencoder  (AUC={auc:.3f}, F1={best_f1:.3f})")
    ax.legend()
    fig.tight_layout()
    hist_path = RESULTS_DIR / f"{TAG}_recon_error_histogram.png"
    fig.savefig(hist_path, dpi=120)
    print(f"\n  Saved histogram to {hist_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(range(1, args.epochs + 1), train_losses, marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE loss")
    ax2.set_title(f"{TAG.upper()} training loss")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    loss_path = RESULTS_DIR / f"{TAG}_training_loss.png"
    fig2.savefig(loss_path, dpi=120)
    print(f"  Saved training-loss curve to {loss_path}")

    # -----------------------------------------------------------------
    # 7. Save numerical summary
    # -----------------------------------------------------------------
    summary_path = RESULTS_DIR / f"{TAG}_autoencoder_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"{TAG.upper()} Autoencoder Anomaly Detection\n")
        f.write("=" * 55 + "\n")
        f.write(f"Train:        {X_train.shape[0]:,} normal samples\n")
        f.write(f"Test:         {X_test.shape[0]:,} "
                f"({(y_test == 1).sum():,} anomalous, {y_test.mean():.2%})\n")
        f.write(f"Input dim:    {input_dim}\n")
        f.write(f"Architecture: {input_dim} -> {h1} -> {h2} -> {lat} -> "
                f"{h2} -> {h1} -> {input_dim}\n")
        f.write(f"Epochs:       {args.epochs}\n")
        f.write("\nMetrics:\n")
        f.write(f"  ROC-AUC:    {auc:.4f}\n")
        f.write(f"  Best F1:    {best_f1:.4f}\n")
        f.write(f"  Precision:  {best_p:.4f}\n")
        f.write(f"  Recall:     {best_r:.4f}\n")
        f.write(f"  Threshold:  {best_thr:.6f}\n")
    print(f"  Saved summary to {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"{TAG.upper()} FINAL:  AUC={auc:.4f}  F1={best_f1:.4f}  "
          f"P={best_p:.4f}  R={best_r:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()