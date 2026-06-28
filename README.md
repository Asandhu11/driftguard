# DriftGuard

**Concept-drift-aware unsupervised anomaly detection in system logs for security.**

Summer 2026 research internship project, supervised by Prof. Yinning Zhang.

## Overview

Static log anomaly detectors degrade silently when system behavior evolves (concept drift). DriftGuard is a three-stage pipeline that detects drift without labels, selectively adapts the model to legitimate drift, and distinguishes gradual drift from abrupt security anomalies.

### Stage 1 - Label-free drift detection
Maximum Mean Discrepancy (MMD) between current and reference autoencoder latent embeddings, with a permutation-test threshold (no labeled anomalies needed).

### Stage 2 - Selective replay-based adaptation
Fine-tune the autoencoder on low-reconstruction-error windows from the drifted region, mixed with a replay buffer from the original training set.

### Stage 3 - Drift vs. attack disambiguation
Per-window features (MMD slope and template entropy) separate drift events from attack events.

## Datasets

- **HDFS_v1** - Hadoop Distributed File System logs (saturated benchmark, non-time-ordered control).
- **BGL** - Blue Gene/L supercomputer logs (time-ordered, 7 months; main dataset for drift experiments).

Both available through the LogHub repository: https://zenodo.org/records/8196385

## Pipeline

Run scripts in this order from the project root:

1. parse_logs.py / parse_bgl.py - raw logs to templates
2. build_sessions.py / build_bgl_features.py - templates to count vectors
3. autoencoder.py - train autoencoder, save encoder and embeddings
4. mmd_drift.py - Stage 1: detect drift
5. mmd_adapt.py - Stage 2: adapt to drift
6. mmd_disambiguate.py - Stage 3: classify drift vs attack
7. stage2_sweep.py / stage2_candidate_sweep.py - hyperparameter sweeps
8. deeplog_lstm.py - DeepLog-style LSTM baseline

## How to reproduce

1. Clone this repo
2. Create Python 3.12 venv: python -m venv venv; .\venv\Scripts\activate
3. Install deps: pip install -r requirements.txt
4. Download HDFS_v1 and BGL from LogHub into data/
5. Run scripts in the order above

## Status

Project in progress. Final report and presentation due July 24, 2026.
