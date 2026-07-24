"""
parse_thunderbird.py
--------------------
Parses Thunderbird_20M.log into fixed-size windows of log-template
sequences, with a time-stamped label per window.

Thunderbird field layout (space-separated) — identical to BGL:
    LABEL  EPOCH  DATE  NODE  MONTH  DAY  TIME  USER@HOST  COMPONENT  MESSAGE...
where LABEL is '-' for normal lines, or a tag for anomalies.

Pipeline:
  1. Stream Thunderbird_20M.log line by line.
  2. Extract label, epoch timestamp, and message text (field 9+).
  3. Parse the message with Drain3 -> template_id.
  4. Group consecutive lines into NON-OVERLAPPING windows of size 100.
  5. Mark a window anomalous if ANY line in it is anomalous.
  6. Save windows to data/thunderbird_windows.csv.

Run from project root:
    python code/parse_thunderbird.py
"""

import csv
import time
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE     = PROJECT_ROOT / "data" / "Thunderbird_20M.log"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "thunderbird_windows.csv"

WINDOW_SIZE = 100   # same convention as BGL / DeepLog / LogBERT


def main():
    if not LOG_FILE.exists():
        print(f"ERROR: Thunderbird_20M.log not found at {LOG_FILE}")
        return

    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th  = 0.4   # LogHub-recommended value for Thunderbird
    config.drain_depth   = 4
    config.masking_instructions = [
        MaskingInstruction(r"(0x)[0-9a-fA-F]+",    "HEX"),
        MaskingInstruction(r"\b[0-9a-fA-F]{8,}\b", "HEX"),
        MaskingInstruction(r"\b\d+\.\d+\.\d+\.\d+\b", "IP"),
        MaskingInstruction(r"\b\d+\b",              "NUM"),
    ]
    miner = TemplateMiner(config=config)

    print(f"Reading: {LOG_FILE}")
    print(f"Window size: {WINDOW_SIZE} lines\n")
    start = time.time()

    n_lines       = 0
    n_anom_lines  = 0
    skipped       = 0
    cur_seq        = []
    cur_anom_count = 0
    cur_start_epoch = None
    window_id      = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE,    "r", encoding="utf-8", errors="ignore") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="")       as f_out:

        writer = csv.writer(f_out)
        writer.writerow([
            "window_id", "start_epoch", "label",
            "n_anom_lines", "length", "template_sequence",
        ])

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=9)
            if len(parts) < 10:
                skipped += 1
                continue

            label_tag   = parts[0]
            epoch       = parts[1]
            message     = parts[9]

            is_anom     = (label_tag != "-")
            template_id = miner.add_log_message(message)["cluster_id"]

            if not cur_seq:
                cur_start_epoch = epoch

            cur_seq.append(template_id)
            if is_anom:
                cur_anom_count += 1
                n_anom_lines   += 1

            n_lines += 1
            if n_lines % 500_000 == 0:
                elapsed = time.time() - start
                rate    = n_lines / elapsed
                pct     = n_anom_lines / n_lines * 100
                eta     = (20_000_000 - n_lines) / rate / 60
                n_tmpl  = len(miner.drain.clusters)
                print(f"  {n_lines/1e6:5.1f}M lines | {rate/1e3:5.0f}k lines/sec | "
                      f"{pct:5.1f}% anomalous | ETA {eta:.1f} min | {n_tmpl} templates")

            if len(cur_seq) >= WINDOW_SIZE:
                writer.writerow([
                    window_id, cur_start_epoch,
                    1 if cur_anom_count > 0 else 0,
                    cur_anom_count, len(cur_seq),
                    " ".join(map(str, cur_seq)),
                ])
                window_id      += 1
                cur_seq         = []
                cur_anom_count  = 0
                cur_start_epoch = None

        if cur_seq:
            writer.writerow([
                window_id, cur_start_epoch,
                1 if cur_anom_count > 0 else 0,
                cur_anom_count, len(cur_seq),
                " ".join(map(str, cur_seq)),
            ])
            window_id += 1

    elapsed     = time.time() - start
    n_templates = len(miner.drain.clusters)

    print(f"\n{'='*60}")
    print(f"Done. Processed {n_lines:,} lines in {elapsed:.1f}s.")
    if skipped:
        print(f"Skipped malformed lines: {skipped:,}")
    print(f"Anomalous lines:  {n_anom_lines:,} ({n_anom_lines/max(n_lines,1):.2%} of lines)")
    print(f"Unique templates: {n_templates}")
    print(f"Total windows:    {window_id:,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    clusters = sorted(miner.drain.clusters, key=lambda c: c.size, reverse=True)
    print("Top 10 most common templates:")
    for i, c in enumerate(clusters[:10], start=1):
        t = c.get_template()
        if len(t) > 80:
            t = t[:77] + "..."
        print(f"  [{i:>2}] {c.size:>9,} lines | {t}")


if __name__ == "__main__":
    main()
