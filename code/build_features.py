"""
build_features.py
-----------------
Converts session template sequences into fixed-length count-vector features
suitable for the autoencoder.

Why count vectors?
  Each session is a sequence of template IDs (variable length). A count vector
  turns it into a fixed-length representation by counting how many times each
  template appears. Order is lost, but for HDFS the mix of templates is
  itself a strong signal (anomalous sessions use unusual template MIXES).
  This is a simpler first baseline; sequence models (DeepLog) come in Week 2.

Train/test split:
  Train = 80% of NORMAL sessions only (the autoencoder learns "what normal looks like")
  Test  = remaining 20% of normals + ALL anomalies (we measure detection)
  This is the standard semi-supervised anomaly detection setup.

Input:
    data/sessions.csv

Output:
    data/features.npz   (X_train, X_test, y_test, template_ids)

Run from project root:
    python code/build_features.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_FILE = PROJECT_ROOT / "data" / "sessions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features.npz"

TRAIN_FRAC = 0.8         # 80% of normals for training
RANDOM_SEED = 42         # fixed for reproducibility


def main():
    start = time.time()

    # -----------------------------------------------------------------
    # 1. Load sessions
    # -----------------------------------------------------------------
    print("Loading sessions...")
    df = pd.read_csv(SESSIONS_FILE)
    print(f"  Loaded {len(df):,} sessions")

    # -----------------------------------------------------------------
    # 2. Parse template_sequence strings ("2 7 1 1 ...") into lists of ints
    # -----------------------------------------------------------------
    print("Parsing template sequences...")
    df["templates"] = df["template_sequence"].apply(
        lambda s: list(map(int, s.split()))
    )

    # -----------------------------------------------------------------
    # 3. Determine the universe of template IDs (these become our columns)
    # -----------------------------------------------------------------
    all_templates = set()
    for seq in df["templates"]:
        all_templates.update(seq)
    template_ids = sorted(all_templates)
    print(f"  Found {len(template_ids)} unique templates "
          f"(IDs {min(template_ids)} to {max(template_ids)})")

    # Mapping: template_id -> column index in the count matrix
    tid_to_col = {tid: i for i, tid in enumerate(template_ids)}

    # -----------------------------------------------------------------
    # 4. Build the count-vector matrix
    # -----------------------------------------------------------------
    print("Building count vectors...")
    n = len(df)
    d = len(template_ids)
    X = np.zeros((n, d), dtype=np.float32)
    for i, seq in enumerate(df["templates"]):
        for tid in seq:
            X[i, tid_to_col[tid]] += 1
    y = df["label"].values.astype(np.int64)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # -----------------------------------------------------------------
    # 5. Split: train = 80% of NORMAL only; test = rest of normals + all anomalies
    # -----------------------------------------------------------------
    print("Splitting into train (normal only) and test (normals + anomalies)...")
    normal_mask = (y == 0)
    X_normal = X[normal_mask]
    X_anomalous = X[~normal_mask]

    X_train, X_test_normal = train_test_split(
        X_normal,
        train_size=TRAIN_FRAC,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    X_test = np.vstack([X_test_normal, X_anomalous])
    y_test = np.concatenate([
        np.zeros(len(X_test_normal), dtype=np.int64),
        np.ones(len(X_anomalous), dtype=np.int64),
    ])
    # Shuffle test set so anomalies don't all sit at the end
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_test))
    X_test = X_test[perm]
    y_test = y_test[perm]

    print(f"  X_train:    {X_train.shape}   (all normal)")
    print(f"  X_test:     {X_test.shape}")
    print(f"  y_test=0:   {(y_test == 0).sum():,}  (normal in test)")
    print(f"  y_test=1:   {(y_test == 1).sum():,}  (anomalous in test, "
          f"{y_test.mean():.2%} of test)")

    # -----------------------------------------------------------------
    # 6. Save
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
    print(f"{'='*60}")
    print(f"Done in {elapsed:.1f}s.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()