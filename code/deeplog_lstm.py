"""
deeplog_lstm.py
---------------
DeepLog-style LSTM next-template prediction model.

Idea (Du et al. 2017):
  Train an LSTM to predict the next template ID given the last k templates.
  Train ONLY on normal sequences. At test time, score a window by the average
  cross-entropy loss of its next-template predictions. Anomalous windows have
  unusual template orderings -> higher loss -> higher anomaly score.

Why this should beat the count-vector autoencoder on BGL:
  Count vectors lose the ORDER of templates. DeepLog uses order directly.

Usage:
    python code/deeplog_lstm.py                       # default: BGL
    python code/deeplog_lstm.py --dataset hdfs        # HDFS sessions
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="bgl", choices=["bgl", "hdfs"])
parser.add_argument("--context", type=int, default=10, help="context window size (k)")
parser.add_argument("--embed-dim", type=int, default=32)
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--train-frac", type=float, default=0.8)
parser.add_argument("--max-train-samples", type=int, default=2_000_000,
                    help="cap on training samples for speed; -1 for all")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TAG = args.dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_sequences():
    """Return (sequences, labels) for the chosen dataset, time- or random-ordered."""
    if TAG == "bgl":
        f = PROJECT_ROOT / "data" / "bgl_windows.csv"
        df = pd.read_csv(f).sort_values("start_epoch").reset_index(drop=True)
        sequences = df["template_sequence"].apply(
            lambda s: list(map(int, s.split()))
        ).tolist()
        labels = df["label"].values
        time_ordered = True
    else:  # hdfs
        f = PROJECT_ROOT / "data" / "sessions.csv"
        df = pd.read_csv(f).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        sequences = df["template_sequence"].apply(
            lambda s: list(map(int, s.split()))
        ).tolist()
        labels = df["label"].values
        time_ordered = False
    return sequences, labels, time_ordered


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
class NextTemplateDataset(Dataset):
    """For each sequence, produce (L - k) (context, target) pairs."""

    def __init__(self, sequences, k):
        self.k = k
        ctx_list = []
        tgt_list = []
        for seq in sequences:
            if len(seq) <= k:
                continue
            for i in range(k, len(seq)):
                ctx_list.append(seq[i - k:i])
                tgt_list.append(seq[i])
        # Convert to tensors at the end for speed
        self.ctx = torch.tensor(ctx_list, dtype=torch.long)
        self.tgt = torch.tensor(tgt_list, dtype=torch.long)

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        return self.ctx[idx], self.tgt[idx]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class DeepLogModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.embed(x)             # (B, k, embed_dim)
        h, _ = self.lstm(e)           # (B, k, hidden_dim)
        logits = self.out(h[:, -1])   # use last hidden state
        return logits


# ---------------------------------------------------------------------
# Per-window scoring
# ---------------------------------------------------------------------
def score_window(model, seq, k, ce_loss):
    """Return mean cross-entropy of next-template predictions in this window."""
    if len(seq) <= k:
        return 0.0
    ctx = torch.tensor([seq[i - k:i] for i in range(k, len(seq))], dtype=torch.long)
    tgt = torch.tensor([seq[i] for i in range(k, len(seq))], dtype=torch.long)
    with torch.no_grad():
        logits = model(ctx)
        loss = ce_loss(logits, tgt).item()
    return loss


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"Dataset: {TAG.upper()}")
    print(f"Context window k = {args.context}")
    print(f"Loading sequences...")
    sequences, labels, time_ordered = load_sequences()
    print(f"  {len(sequences):,} sequences loaded")
    print(f"  Time-ordered split: {time_ordered}")

    # Vocabulary size (template IDs are 1..max)
    max_tid = max(max(s) for s in sequences)
    vocab_size = max_tid + 1
    print(f"  Vocab size: {vocab_size}")

    # Split
    n = len(sequences)
    split_idx = int(n * args.train_frac)
    if time_ordered:
        train_idx = list(range(split_idx))
        test_idx = list(range(split_idx, n))
    else:
        # already shuffled in load_sequences for HDFS
        train_idx = list(range(split_idx))
        test_idx = list(range(split_idx, n))

    train_seqs = [sequences[i] for i in train_idx if labels[i] == 0]
    test_seqs = [sequences[i] for i in test_idx]
    test_labels = labels[test_idx]
    print(f"  Train (normal only): {len(train_seqs):,}")
    print(f"  Test:  {len(test_seqs):,}  "
          f"({(test_labels == 1).sum():,} anomalous)")

    # Build training dataset
    print(f"\nBuilding (context, target) pairs...")
    train_ds = NextTemplateDataset(train_seqs, args.context)
    n_pairs = len(train_ds)
    print(f"  {n_pairs:,} training pairs")

    # Optionally subsample for speed
    if args.max_train_samples > 0 and n_pairs > args.max_train_samples:
        keep = args.max_train_samples
        idx = torch.randperm(n_pairs)[:keep]
        train_ds.ctx = train_ds.ctx[idx]
        train_ds.tgt = train_ds.tgt[idx]
        print(f"  Subsampled to {keep:,} for speed")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    model = DeepLogModel(vocab_size, args.embed_dim, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_train = nn.CrossEntropyLoss()
    ce_eval = nn.CrossEntropyLoss(reduction="mean")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Train
    print(f"\nTraining {args.epochs} epochs...")
    losses = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for ctx, tgt in loader:
            optimizer.zero_grad()
            logits = model(ctx)
            loss = ce_train(logits, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / n_batches
        losses.append(avg)
        print(f"  Epoch {epoch}/{args.epochs}  loss = {avg:.4f}")
    train_time = time.time() - start
    print(f"Training done in {train_time:.1f}s.")

    # Score each test window
    print(f"\nScoring {len(test_seqs):,} test windows...")
    model.eval()
    scores = np.zeros(len(test_seqs), dtype=np.float32)
    t0 = time.time()
    for i, seq in enumerate(test_seqs):
        scores[i] = score_window(model, seq, args.context, ce_eval)
        if (i + 1) % 1000 == 0:
            print(f"  Scored {i + 1:,}/{len(test_seqs):,}")
    print(f"Scoring done in {time.time() - t0:.1f}s.")

    # Metrics
    auc = roc_auc_score(test_labels, scores)
    precision, recall, thresholds = precision_recall_curve(test_labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1))
    best_f1 = f1[best_idx]
    best_p = precision[best_idx]
    best_r = recall[best_idx]
    best_thr = thresholds[min(best_idx, len(thresholds) - 1)]

    print(f"\n{'=' * 60}")
    print(f"{TAG.upper()} DeepLog-LSTM:  AUC={auc:.4f}  F1={best_f1:.4f}  "
          f"P={best_p:.4f}  R={best_r:.4f}")
    print(f"{'=' * 60}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    upper = float(np.percentile(scores, 99))
    bins = np.linspace(0, upper, 80)
    ax.hist(scores[test_labels == 0], bins=bins, alpha=0.6,
            label="Normal", color="steelblue", density=True)
    ax.hist(scores[test_labels == 1], bins=bins, alpha=0.6,
            label="Anomalous", color="crimson", density=True)
    ax.axvline(best_thr, color="black", linestyle="--",
               label=f"Best-F1 threshold = {best_thr:.4f}")
    ax.set_xlabel("Mean next-template cross-entropy")
    ax.set_ylabel("Density")
    ax.set_title(f"{TAG.upper()} DeepLog-LSTM  (AUC={auc:.3f}, F1={best_f1:.3f})")
    ax.legend()
    fig.tight_layout()
    p = RESULTS_DIR / f"{TAG}_deeplog_histogram.png"
    fig.savefig(p, dpi=120)
    print(f"\nSaved histogram to {p}")

    # Summary
    with open(RESULTS_DIR / f"{TAG}_deeplog_results.txt", "w") as f:
        f.write(f"{TAG.upper()} DeepLog-style LSTM\n")
        f.write("=" * 55 + "\n")
        f.write(f"Context k:   {args.context}\n")
        f.write(f"Embed dim:   {args.embed_dim}\n")
        f.write(f"Hidden dim:  {args.hidden_dim}\n")
        f.write(f"Vocab size:  {vocab_size}\n")
        f.write(f"Train pairs: {len(train_ds):,}\n")
        f.write(f"Test seqs:   {len(test_seqs):,} "
                f"({(test_labels == 1).sum():,} anomalous)\n")
        f.write(f"Epochs:      {args.epochs}\n")
        f.write(f"Train time:  {train_time:.1f}s\n\n")
        f.write("Metrics:\n")
        f.write(f"  ROC-AUC:    {auc:.4f}\n")
        f.write(f"  Best F1:    {best_f1:.4f}\n")
        f.write(f"  Precision:  {best_p:.4f}\n")
        f.write(f"  Recall:     {best_r:.4f}\n")


if __name__ == "__main__":
    main()