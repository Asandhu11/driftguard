# DriftGuard

**Concept-drift-aware unsupervised anomaly detection in system logs.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LogHub](https://img.shields.io/badge/datasets-LogHub-orange.svg)](https://zenodo.org/records/8196385)

> Summer 2026 Research Internship · University of West Georgia · Supervised by Prof. Yinning Zhang

---

## What is DriftGuard?

Static log anomaly detectors degrade silently when system behavior evolves — a problem called **concept drift**. DriftGuard is a three-stage label-free pipeline that:

| Stage | What it does | How |
|---|---|---|
| **1 · Detect** | Finds when the log distribution has shifted | MMD on autoencoder latent embeddings, permutation-test threshold |
| **2 · Adapt** | Updates the model to new normal behavior | Selective pseudo-normal replay fine-tuning |
| **3 · Disambiguate** | Tells drift apart from attacks | Template entropy + MMD slope per window |

No labels required at any stage — including threshold calibration.

---

## Results

Evaluated on three [LogHub](https://zenodo.org/records/8196385) benchmark datasets:

### Baseline anomaly detection (autoencoder, no labels)

| Dataset | AUC | F1 | Notes |
|---|---|---|---|
| BGL | 0.7184 | 0.3539 | 8.66% anomalous test windows; F1 low due to class imbalance |
| **Thunderbird** | **0.8412** | **0.8684** | Zero Thunderbird-specific tuning — cross-dataset transfer |

### Stage 1 — Label-free drift detection

| Dataset | Alarms | First alarm | Corr(MMD, anomaly) |
|---|---|---|---|
| HDFS (control) | **0 / 90** | never | — |
| BGL | 82 / 90 | test idx 300 | +0.310 |
| Thunderbird | 395 / 395 | test idx 0 | +0.750 |

HDFS fires zero alarms — the pipeline does not hallucinate drift on stable systems.

### Stage 3 — Drift vs. attack disambiguation

| Dataset | Corr(entropy, anomaly) | Direction |
|---|---|---|
| BGL | +0.816 | High entropy → anomaly (broadcast-divergent failures) |
| Thunderbird | −0.767 | Low entropy → anomaly (repetitive-convergent failures) |

---

## Repository Structure

```
driftguard/
├── code/
│   ├── parse_logs.py               # HDFS log parsing (Drain3)
│   ├── parse_bgl.py                # BGL log parsing (Drain3)
│   ├── parse_thunderbird.py        # Thunderbird log parsing (Drain3)
│   ├── build_sessions.py           # HDFS session feature extraction
│   ├── build_bgl_features.py       # BGL count-vector windows
│   ├── build_thunderbird_features.py # Thunderbird count-vector windows
│   ├── autoencoder.py              # Train autoencoder, save embeddings
│   ├── mmd_drift.py                # Stage 1: MMD drift detection
│   ├── mmd_adapt.py                # Stage 2: selective replay adaptation
│   ├── mmd_disambiguate.py         # Stage 3: drift vs. attack
│   ├── deeplog_lstm.py             # DeepLog LSTM baseline
│   ├── stage2_sweep.py             # Stage 2 hyperparameter sweep
│   └── stage2_candidate_sweep.py   # Stage 2 candidate threshold sweep
├── results/                        # Generated plots and summary files
├── REPORT.md                       # Full project report
├── DriftGuard_IEEE.md              # IEEE-format paper draft
├── driftguard_slides.pptx          # Presentation slides
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Asandhu11/driftguard
cd driftguard
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Download datasets

Download from [LogHub](https://zenodo.org/records/8196385) and place in `data/`:

```
data/
  HDFS_v1/HDFS.log
  BGL/BGL.log
  Thunderbird_20M.log          # first 20M lines of Thunderbird.log
```

To extract the Thunderbird subset on Windows:
```powershell
Get-Content data\Thunderbird.log -TotalCount 20000000 | Set-Content data\Thunderbird_20M.log
```

### 3. Run the pipeline

**BGL (main experiment):**
```bash
python code/parse_bgl.py
python code/build_bgl_features.py
python code/autoencoder.py --features bgl_features.npz --tag bgl
python code/mmd_drift.py --tag bgl
python code/mmd_adapt.py --tag bgl
python code/mmd_disambiguate.py --tag bgl
```

**Thunderbird (cross-dataset validation):**
```bash
python code/parse_thunderbird.py
python code/build_thunderbird_features.py
python code/autoencoder.py --features thunderbird_features.npz --tag thunderbird
python code/mmd_drift.py --tag thunderbird
python code/mmd_adapt.py --tag thunderbird
python code/mmd_disambiguate.py --tag thunderbird
```

**HDFS (drift-free control):**
```bash
python code/parse_logs.py
python code/build_sessions.py
python code/autoencoder.py --features features.npz --tag hdfs
python code/mmd_drift.py --tag hdfs
```

---

## Key Findings

1. **Zero-shot cross-dataset transfer works.** AUC 0.84 on Thunderbird with no Thunderbird-specific tuning.
2. **MMD tracks anomaly activity without labels.** Corr(MMD, anomaly fraction) = +0.31 (BGL) and +0.75 (Thunderbird).
3. **HDFS control fires zero alarms.** Confirms the permutation-test threshold does not hallucinate drift.
4. **Stage 2 has a defined failure boundary.** Pseudo-normal selection degrades when drift-region anomaly density exceeds ~10%. Operational rule: skip adaptation above this threshold.
5. **Template entropy is a reliable disambiguation feature** — but its sign depends on failure-mode character (broadcast-divergent vs. repetitive-convergent).

---

## How the Pipeline Works

A full explanation with code examples, annotated diagrams, and results tables is in [`REPORT.md`](REPORT.md).

An IEEE-format paper draft is in [`DriftGuard_IEEE.md`](DriftGuard_IEEE.md).

---

## Citation

If you use this code or findings in your work, please cite:

```
@misc{sandhu2026driftguard,
  title  = {DriftGuard: Concept-Drift-Aware Unsupervised Anomaly Detection in System Logs},
  author = {Sandhu, Amninder Singh},
  year   = {2026},
  url    = {https://github.com/Asandhu11/driftguard},
  note   = {Supervised by Prof. Yinning Zhang, University of West Georgia}
}
```

---

## Datasets

All datasets are from the LogHub repository:

> J. Zhu et al., "Loghub: A Large Collection of System Log Datasets for AI-Driven Log Analytics," ISSRE 2023. https://zenodo.org/records/8196385

---

## License

MIT License — see [LICENSE](LICENSE) for details.
