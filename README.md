# DriftGuard

**Concept-drift-aware unsupervised anomaly detection in system logs for security.**

Summer 2026 research internship project, supervised by Prof. Yinning Zhang.

## Overview

Static log anomaly detectors degrade silently when system behavior evolves
(concept drift). DriftGuard is a three-stage pipeline that detects drift
without labels, selectively adapts the model to legitimate drift, and
distinguishes gradual drift from abrupt security anomalies — so a system
can respond correctly (adapt vs. alert).

### Stage 1 — Label-free drift detection
Maximum Mean Discrepancy (MMD) between current and reference autoencoder
latent embeddings, with a permutation-test threshold (no labeled anomalies
needed).

### Stage 2 — Selective replay-based adaptation
Fine-tune the autoencoder on low-reconstruction-error windows from the
drifted region, mixed with a replay buffer from the original training set.

### Stage 3 — Drift vs. attack disambiguation
Per-window features (MMD slope and template entropy) separate drift events
from attack events on the same MMD curve.

## Datasets

- **HDFS_v1** — Hadoop Distributed File System logs. Used as a saturated
  benchmark and a non-time-ordered control.
- **BGL** — Blue Gene/L supercomputer logs (time-ordered, 7 months). Main
  dataset for the drift experiments.

Both available through the LogHub repository.

## Pipeline

parse_logs.py / parse_bgl.py    → raw logs → templates

build_sessions.py / build_bgl_features.py → templates → count vectors

autoencoder.py                  → train AE, save encoder + embeddings

mmd_drift.py                    → Stage 1: detect drift

mmd_adapt.py                    → Stage 2: adapt

mmd_disambiguate.py             → Stage 3: classify drift vs attack

stage2_sweep.py

stage2_candidate_sweep.py       → hyperparameter sweeps

deeplog_lstm.py                 → DeepLog-style LSTM baseline

## How to reproduce

1. Clone this repo.
2. Create and activate a Python 3.12 virtual environment:
 python -m venv venv

.\venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt
4. Download HDFS_v1 and BGL from LogHub
   (https://zenodo.org/records/8196385) into `data/`.
5. Run the pipeline scripts in order (see `pipeline` above).

## Results

See `results/` for plots and metrics produced by each stage.

## Status

Project in progress. Final report and presentation due July 24, 2026.
