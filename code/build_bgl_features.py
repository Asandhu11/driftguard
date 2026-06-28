"""
build_bgl_features.py
---------------------
Converts BGL windows into count-vector features with a TIME-ORDERED split.

Key difference from HDFS:
  HDFS was random-split because it has no time ordering.
  BGL is time-ordered, so we use the FIRST 80% (chronologically) for training
  and the LAST 20% for testing. This is more realistic and sets up the drift
  experiments in Week 3.

Train set: first 80% of windows, NORMAL only.
Test set:  last 20% of windows, both normal and anomalous.

Input:
    data/bgl_windows.csv

Output:
    data/bgl_features.npz   (X_train, X_test, y_test, template_ids)

Run from project root:
    python code/build_bgl_features.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_FILE = PROJECT_ROOT / "data" / "bgl_windows.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "bgl_features.npz"

TRAIN_FRAC = 0.8


def main():
    start = time.time()

    # -----------------------------------------------------------------
    # 1. Load BGL windows
    # -----------------------------------------------------------------
    print("Loading BGL windows...")
    df = pd.read_csv(WINDOWS_FILE)
    print(f"  Loaded {len(df):,} windows")
    print(f"  Overall window-level anomaly rate: {df['label'].mean():.2%}")

    # -----------------------------------------------------------------
    # 2. Sort by start_epoch (should already be in order, but enforce it)
    # -----------------------------------------------------------------
    df = df.sort_values("start_epoch").reset_index(drop=True)

    # -----------------------------------------------------------------
    # 3. Parse template_sequence strings into lists of ints
    # -----------------------------------------------------------------
    print("Parsing template sequences...")
    df["templates"] = df["template_sequence"].apply(
        lambda s: list(map(int, s.split()))
    )

    # -----------------------------------------------------------------
    # 4. Determine template universe (columns of feature matrix)
    # -----------------------------------------------------------------
    all_templates = set()
    for seq in df["templates"]:
        all_templates.update(seq)
    template_ids = sorted(all_templates)
    print(f"  Found {len(template_ids)} unique templates "
          f"(IDs {min(template_ids)} to {max(template_ids)})")

    tid_to_col = {tid: i for i, tid in enumerate(template_ids)}

    # -----------------------------------------------------------------
    # 5. Build count-vector matrix
    # -----------------------------------------------------------------
    print("Building count vectors...")
    n = len(df)
    d = len(template_ids)
    X = np.zeros((n, d), dtype=np.float32)
    for i, seq in enumerate(df["templates"]):
        for tid in seq:
            X[i, tid_to_col[tid]] += 1
    y = df["label"].values.astype(np.int64)
    print(f"  X shape: {X.shape}")

    # -----------------------------------------------------------------
    # 6. TIME-ORDERED split (the critical difference from HDFS)
    # -----------------------------------------------------------------
    split_idx = int(n * TRAIN_FRAC)
    print(f"\nTime-ordered split at index {split_idx:,} "
          f"({TRAIN_FRAC:.0%} / {1-TRAIN_FRAC:.0%}):")

    X_train_pool = X[:split_idx]
    y_train_pool = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # -----------------------------------------------------------------
    # 7. Training pool -> keep only NORMAL windows
    #    (semi-supervised setup: AE learns "normal" only)
    # -----------------------------------------------------------------
    normal_mask = (y_train_pool == 0)
    X_train = X_train_pool[normal_mask]

    print(f"  Training pool (first {TRAIN_FRAC:.0%}): {split_idx:,} windows")
    print(f"  -> X_train (normals only):              {X_train.shape}")
    print(f"     (dropped {(~normal_mask).sum():,} anomalous "
          f"from training pool)")
    print(f"  Test set (last {1-TRAIN_FRAC:.0%}):     {X_test.shape}")
    print(f"     Normal in test:     {(y_test == 0).sum():,}")
    print(f"     Anomalous in test:  {(y_test == 1).sum():,} "
          f"({y_test.mean():.2%})")

    # -----------------------------------------------------------------
    # 8. Save
    # -----------------------------------------------------------------
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        template_ids=np.array(template_ids),
    )
    elapsed = time.time() - start
    print(f"\nSaved features to {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    print(f"Done in {elapsed:.1f}s.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()