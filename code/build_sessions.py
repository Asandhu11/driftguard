"""
build_sessions.py
-----------------
Groups parsed log lines into block-level sessions and joins with anomaly labels.

Input:
    data/parsed_sample.csv                         (from parse_logs.py)
    data/HDFS_v1/preprocessed/anomaly_label.csv    (from the dataset)

Output:
    data/sessions.csv -- one row per block_id with its template sequence and label

Run from project root:
    python code/build_sessions.py
"""

import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_FILE = PROJECT_ROOT / "data" / "parsed_sample.csv"
LABEL_FILE = PROJECT_ROOT / "data" / "HDFS_v1" / "preprocessed" / "anomaly_label.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "sessions.csv"


def main():
    start = time.time()

    # -----------------------------------------------------------------
    # 1. Load parsed log lines. We only need 3 columns to save memory.
    # -----------------------------------------------------------------
    print("Loading parsed log lines...")
    df = pd.read_csv(
        PARSED_FILE,
        usecols=["line_number", "template_id", "block_id"],
    )
    print(f"  Loaded {len(df):,} lines")

    # -----------------------------------------------------------------
    # 2. Drop lines that didn't have a block_id.
    # -----------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["block_id"])
    df = df[df["block_id"].astype(str) != ""]
    print(f"  Dropped {before - len(df):,} lines without a block_id")
    print(f"  Keeping {len(df):,} lines")

    # -----------------------------------------------------------------
    # 3. Sort by line_number so each session keeps its original order.
    # -----------------------------------------------------------------
    df = df.sort_values("line_number")

    # -----------------------------------------------------------------
    # 4. Group by block_id, collect the template sequence as a string.
    # -----------------------------------------------------------------
    print("\nGrouping into sessions...")
    sessions = (
        df.groupby("block_id")["template_id"]
          .apply(lambda x: " ".join(map(str, x)))
          .reset_index()
    )
    sessions.columns = ["block_id", "template_sequence"]
    sessions["session_length"] = sessions["template_sequence"].str.split().str.len()
    print(f"  Found {len(sessions):,} unique sessions")

    # -----------------------------------------------------------------
    # 5. Load anomaly labels.
    # -----------------------------------------------------------------
    print("\nLoading anomaly labels...")
    labels = pd.read_csv(LABEL_FILE)
    labels.columns = ["block_id", "label_str"]
    labels["label"] = (labels["label_str"] == "Anomaly").astype(int)
    print(f"  Loaded {len(labels):,} labels")
    print(f"  Anomaly rate in labels: {labels['label'].mean():.2%}")

    # -----------------------------------------------------------------
    # 6. Join sessions with labels.
    # -----------------------------------------------------------------
    print("\nJoining sessions with labels...")
    merged = sessions.merge(
        labels[["block_id", "label"]],
        on="block_id",
        how="left",
    )
    unlabeled = merged["label"].isna().sum()
    if unlabeled > 0:
        print(f"  WARNING: {unlabeled:,} sessions had no label (dropping them)")
        merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)

    print(f"  Final sessions: {len(merged):,}")
    print(f"  Normal:     {(merged['label']==0).sum():,}")
    print(f"  Anomalous:  {(merged['label']==1).sum():,} "
          f"({merged['label'].mean():.2%})")

    # -----------------------------------------------------------------
    # 7. Save (reorder columns to put label up front for readability).
    # -----------------------------------------------------------------
    merged = merged[["block_id", "label", "session_length", "template_sequence"]]
    print(f"\nSaving to {OUTPUT_FILE}...")
    merged.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s.")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # 8. Print session-length stats. Anomalous sessions often have
    #    different length distributions than normal ones.
    # -----------------------------------------------------------------
    print("Session length overall:")
    print(merged["session_length"].describe().to_string())
    print("\nSession length: Normal vs Anomalous")
    print(merged.groupby("label")["session_length"].describe().to_string())


if __name__ == "__main__":
    main()