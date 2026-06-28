"""
parse_bgl.py
------------
Parses BGL (Blue Gene/L) supercomputer logs into fixed-size windows of
log-template sequences, with a time-stamped label per window.

BGL format (space-separated):
    LABEL  EPOCH  DATE  NODE1  DATETIME  NODE2  SYSTEM  COMPONENT  LEVEL  MESSAGE...
where LABEL is '-' for normal lines, or an alert tag like 'KERNDTLB' for anomalies.

Pipeline:
  1. Stream BGL.log line by line.
  2. Extract label, epoch timestamp, and message text.
  3. Parse the message with Drain3 -> template_id.
  4. Group consecutive lines into NON-OVERLAPPING windows of size 100
     (standard practice; matches DeepLog and LogBERT BGL setup).
  5. Mark a window as anomalous if ANY line in it is anomalous.
  6. Save windows to data/bgl_windows.csv.

Why fixed-size windows (not sessions)?
  Unlike HDFS, BGL has no per-session ID. It is a continuous stream of
  log lines with a real timestamp on each one. Fixed windows give every
  window the same shape (length 100), which is what the autoencoder needs.

Run from project root:
    python code/parse_bgl.py
"""

import csv
import time
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "BGL" / "BGL.log"
OUTPUT_FILE = PROJECT_ROOT / "data" / "bgl_windows.csv"

WINDOW_SIZE = 100   # follows DeepLog/LogBERT convention for BGL


def main():
    if not LOG_FILE.exists():
        print(f"ERROR: BGL.log not found at {LOG_FILE}")
        return

    # Configure Drain3 (default settings work well on BGL).
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)

    print(f"Reading: {LOG_FILE}")
    print(f"Window size: {WINDOW_SIZE} lines\n")
    start = time.time()

    # Counters.
    n_lines = 0
    n_anom_lines = 0
    skipped = 0

    # Current-window accumulator.
    cur_seq = []                # list of template_ids
    cur_anom_count = 0
    cur_start_epoch = None
    window_id = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out)
        writer.writerow([
            "window_id",
            "start_epoch",
            "label",
            "n_anom_lines",
            "length",
            "template_sequence",
        ])

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # Split into max 10 parts -- everything after field 9 is the message.
            parts = line.split(maxsplit=9)
            if len(parts) < 10:
                skipped += 1
                continue

            label_tag = parts[0]
            epoch = parts[1]
            message = parts[9]

            is_anom = (label_tag != "-")
            template_id = miner.add_log_message(message)["cluster_id"]

            # If this is the first line of a new window, record start_epoch.
            if not cur_seq:
                cur_start_epoch = epoch

            cur_seq.append(template_id)
            if is_anom:
                cur_anom_count += 1
                n_anom_lines += 1

            n_lines += 1
            if n_lines % 500_000 == 0:
                elapsed = time.time() - start
                rate = n_lines / elapsed
                print(f"  Processed {n_lines:>9,} lines  "
                      f"({rate:,.0f} lines/sec)")

            # Flush a completed window.
            if len(cur_seq) >= WINDOW_SIZE:
                writer.writerow([
                    window_id,
                    cur_start_epoch,
                    1 if cur_anom_count > 0 else 0,
                    cur_anom_count,
                    len(cur_seq),
                    " ".join(map(str, cur_seq)),
                ])
                window_id += 1
                cur_seq = []
                cur_anom_count = 0
                cur_start_epoch = None

        # Trailing partial window (if any).
        if cur_seq:
            writer.writerow([
                window_id,
                cur_start_epoch,
                1 if cur_anom_count > 0 else 0,
                cur_anom_count,
                len(cur_seq),
                " ".join(map(str, cur_seq)),
            ])
            window_id += 1

    elapsed = time.time() - start
    n_templates = len(miner.drain.clusters)

    print(f"\n{'='*60}")
    print(f"Done. Processed {n_lines:,} lines in {elapsed:.1f}s.")
    if skipped:
        print(f"Skipped malformed lines: {skipped:,}")
    print(f"Anomalous lines:  {n_anom_lines:,} "
          f"({n_anom_lines/max(n_lines,1):.2%} of lines)")
    print(f"Unique templates: {n_templates}")
    print(f"Total windows:    {window_id:,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    # Top 10 most common templates.
    clusters = sorted(miner.drain.clusters, key=lambda c: c.size, reverse=True)
    print("Top 10 most common templates:")
    for i, c in enumerate(clusters[:10], start=1):
        t = c.get_template()
        if len(t) > 80:
            t = t[:77] + "..."
        print(f"  [{i:>2}] {c.size:>9,} lines | {t}")


if __name__ == "__main__":
    main()