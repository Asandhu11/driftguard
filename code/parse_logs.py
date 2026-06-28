"""
parse_logs.py
-------------
Parses raw HDFS log lines into structured templates using Drain3.

This is Week 1, Day 3-4 of the DriftGuard internship project.

What it does:
  1. Reads the first 100,000 lines of HDFS.log
  2. Feeds each line to Drain3, which extracts a template
     (replacing variable parts like IPs and block IDs with <*>)
  3. Saves one row per log line with its assigned template_id
  4. Prints summary stats

Run from VS Code terminal:
    python code/parse_logs.py
"""

import csv
import time
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# ---------------------------------------------------------------------
# Paths — relative to the driftguard project root.
# Path objects work cross-platform (no Windows vs Linux slash issues).
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level up from /code
LOG_FILE = PROJECT_ROOT / "data" / "HDFS_v1" / "HDFS.log"
OUTPUT_FILE = PROJECT_ROOT / "data" / "parsed_sample.csv"

# How many lines to process. Start small (100k) to verify the script works.
# We will scale up to all ~11 million lines later.
MAX_LINES = 12_000_000


def main():
    # -----------------------------------------------------------------
    # 1. Configure Drain3.
    # -----------------------------------------------------------------
    # Drain works by building a tree of log templates. Default config
    # is fine for HDFS. We disable masking of numbers because HDFS
    # block IDs (blk_-1608...) are the key identifier we need preserved
    # for later session grouping.
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    template_miner = TemplateMiner(config=config)

    # -----------------------------------------------------------------
    # 2. Check the log file exists.
    # -----------------------------------------------------------------
    if not LOG_FILE.exists():
        print(f"ERROR: Log file not found at {LOG_FILE}")
        print("Make sure HDFS_v1.zip was extracted into the data/ folder.")
        return

    print(f"Reading from: {LOG_FILE}")
    print(f"Will process up to {MAX_LINES:,} lines.")
    print("Starting parsing...\n")

    # -----------------------------------------------------------------
    # 3. Open the log file and the output CSV. Stream line-by-line
    #    so we don't load the whole 1.5 GB file into memory.
    # -----------------------------------------------------------------
    start_time = time.time()
    line_count = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["line_number", "template_id", "template", "block_id"])

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # ---------------------------------------------------------
            # The HDFS log format is:
            #   081109 203518 143 INFO dfs.X: <message>
            # We strip the first 5 fields (date, time, pid, level, component)
            # because they are noise — only the message content matters
            # for template extraction.
            # ---------------------------------------------------------
            parts = line.split(maxsplit=5)
            if len(parts) < 6:
                continue
            message = parts[5]

            # Feed the message to Drain3
            result = template_miner.add_log_message(message)
            template_id = result["cluster_id"]
            template = result["template_mined"]

            # ---------------------------------------------------------
            # Extract the block ID from the message. Every HDFS log line
            # contains exactly one block ID like blk_-1608999687919862906.
            # We need it so we can later group lines into sessions.
            # ---------------------------------------------------------
            block_id = ""
            for token in message.split():
                if token.startswith("blk_"):
                    # Strip trailing punctuation (e.g. "blk_123.")
                    block_id = token.rstrip(".,:;")
                    break

            writer.writerow([line_count, template_id, template, block_id])
            line_count += 1

            # Progress indicator every 10k lines
            if line_count % 10_000 == 0:
                elapsed = time.time() - start_time
                rate = line_count / elapsed
                print(f"  Processed {line_count:>7,} lines "
                      f"({rate:,.0f} lines/sec)")

            if line_count >= MAX_LINES:
                break

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------
    # 4. Print summary.
    # -----------------------------------------------------------------
    clusters = template_miner.drain.clusters
    print(f"\n{'=' * 60}")
    print(f"Done. Processed {line_count:,} lines in {elapsed:.1f}s.")
    print(f"Found {len(clusters)} unique templates.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")

    # Print top 10 most common templates
    sorted_clusters = sorted(clusters, key=lambda c: c.size, reverse=True)
    print(f"Top 10 most common templates:\n")
    for i, cluster in enumerate(sorted_clusters[:10], start=1):
        template_text = cluster.get_template()
        if len(template_text) > 90:
            template_text = template_text[:87] + "..."
        print(f"  [{i:>2}] {cluster.size:>7,} lines | {template_text}")


if __name__ == "__main__":
    main()
