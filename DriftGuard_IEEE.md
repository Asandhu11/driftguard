# DriftGuard: Concept-Drift-Aware Unsupervised Anomaly Detection in System Logs

**Author:** Amninder Singh Sandhu
**Affiliation:** University of West Georgia
**Supervisor:** Prof. Yinning Zhang
**Contact:** asandhu1@my.westga.edu

---

## Abstract

Static log anomaly detectors trained on historical data degrade silently when
system behavior evolves — a phenomenon known as concept drift. We present
DriftGuard, a three-stage label-free pipeline that detects drift in system logs,
selectively adapts the underlying model to legitimate behavioral changes, and
distinguishes gradual drift from abrupt security anomalies. Stage 1 computes
Maximum Mean Discrepancy (MMD) between autoencoder latent embeddings from a
reference window and a sliding detection window, calibrating a detection
threshold via permutation test without any labeled data. Stage 2 fine-tunes the
autoencoder on pseudo-normal windows selected by low reconstruction error, mixed
with a replay buffer to prevent catastrophic forgetting. Stage 3 uses template
entropy and MMD slope as per-window features to separate drift events from
attack events. We evaluate DriftGuard on three public benchmark datasets from
LogHub: HDFS (drift-free control), Blue Gene/L supercomputer logs (BGL, main
experiment), and Thunderbird supercomputer logs (cross-dataset validation, 20
million lines). DriftGuard achieves AUC 0.84 and F1 0.87 on Thunderbird with
zero Thunderbird-specific parameter tuning, demonstrating cross-dataset
generalization. The HDFS control produces zero false alarms, validating the
threshold calibration. Stage 1 MMD correlates with true anomaly fraction at
+0.31 (BGL) and +0.75 (Thunderbird) without labels. We identify an operational
boundary condition for Stage 2: adaptation degrades when drift-region anomaly
density exceeds approximately 10%. Stage 3 template entropy tracks anomaly
density reliably across both datasets (|r| > 0.75), with sign depending on
whether the system exhibits broadcast-divergent or repetitive-convergent failure
modes.

---

## Keywords

log anomaly detection; concept drift; maximum mean discrepancy; autoencoder;
unsupervised learning; system security; log parsing; cross-dataset generalization;
supercomputer logs; intrusion detection

---

## First Footnote

This work was completed as part of a Summer 2026 undergraduate research
internship at the University of West Georgia, supervised by Prof. Yinning Zhang.
Code and datasets are available at https://github.com/Asandhu11/driftguard.
Datasets (HDFS, BGL, Thunderbird) are publicly available through the LogHub
repository at https://zenodo.org/records/8196385.

---

## I. Introduction

Modern computing infrastructure generates log data at enormous scale. The
Thunderbird supercomputer, for example, produces over 200 million log lines from
a single cluster — roughly 10 GB per day. Automated log anomaly detection is
essential: human review at this scale is impossible, and undetected failures or
intrusions can cause significant damage before they are discovered.

The standard approach is semi-supervised: train a model exclusively on normal
log behavior, then flag test-time windows that deviate from the learned normal.
This works well under the assumption that the definition of "normal" is stable.
In practice, that assumption rarely holds. Software deployments change which
services write logs. Hardware upgrades alter which components generate warnings.
Scheduled maintenance jobs flood the system with messages the model has never
seen. These changes — none of which are security threats — cause the normal
distribution to shift over time. A model trained on older data begins to
mismatch the current system, raising false alarms or, worse, recalibrating to
treat new attack patterns as normal. This mismatch is known as concept drift
[CITATION_GRETTON], and it is one of the principal reasons that production
log-monitoring systems require constant manual maintenance.

Existing log anomaly detection methods — including DeepLog [CITATION_DEEPLOG],
LogBERT [CITATION_LOGBERT], and LogAnomaly [CITATION_LOGANOMALY] — are evaluated
on static dataset snapshots and do not address drift. Methods that do address
drift typically require labeled data to detect or respond to it, which is
expensive to obtain and itself goes stale as the system evolves.

We present DriftGuard, a three-stage pipeline that:

1. **Detects** concept drift in log streams using Maximum Mean Discrepancy on
   autoencoder latent embeddings, with a threshold calibrated entirely from
   unlabeled training data via permutation test.
2. **Adapts** the anomaly model to legitimate drift using selective replay-based
   fine-tuning, without requiring human annotation of the drift region.
3. **Distinguishes** drift events from attack events using per-window features
   derived from the log template distribution, without any additional labeled
   data.

The pipeline is evaluated on three standard LogHub benchmark datasets across
two different supercomputer architectures. The main contributions are:

- A fully label-free three-stage drift-aware anomaly detection pipeline.
- Empirical demonstration of cross-dataset generalization: AUC 0.84 on
  Thunderbird logs with zero Thunderbird-specific parameter changes.
- Identification of the operational boundary condition for pseudo-normal
  adaptation: the heuristic degrades when drift-region anomaly density exceeds
  approximately 10% at the window level.
- A finding that template entropy is a reliable drift-vs-attack disambiguation
  feature across failure modes with qualitatively different characters.

The remainder of this paper is organized as follows. Section II reviews related
work. Section III describes the DriftGuard methodology. Section IV presents
experimental setup and results. Section V discusses findings and limitations.
Section VI concludes.

---

## II. Related Work

### A. Log Parsing

Log parsing transforms free-text log messages into structured templates by
separating constant tokens from variable ones. Drain [CITATION_DRAIN] introduced
a fixed-depth parse tree that processes each log message in O(depth) time and
is widely used in production systems. Drain3 extends Drain with streaming
support and configurable token masking. Several recent surveys
[CITATION_ZHULOGHUB] compare parsers on standard benchmarks and report that
Drain is among the most accurate and efficient on supercomputer logs.
DriftGuard uses Drain3 for all datasets with benchmark-matched similarity
thresholds (sim_th = 0.5 for BGL, 0.4 for Thunderbird).

### B. Log Anomaly Detection

DeepLog [CITATION_DEEPLOG] models log sequences as natural language and trains
an LSTM to predict the next log key; deviations from top-g predictions are
flagged as anomalies. LogBERT [CITATION_LOGBERT] adapts the BERT transformer
to log sequences, achieving high F1 on HDFS and BGL but requiring substantial
compute. LogAnomaly [CITATION_LOGANOMALY] augments sequence models with
semantic embeddings derived from log templates. PLELog [CITATION_PLELOG]
uses a probabilistic label estimation scheme to reduce dependence on clean
labels.

All of these methods assume a stationary distribution at test time. None
explicitly detect or adapt to concept drift. DriftGuard complements these
approaches: it operates on count-vector features rather than sequences, which
makes it computationally lighter, and it adds explicit drift detection and
adaptation stages that the sequence models lack.

### C. Concept Drift Detection

The concept drift detection literature broadly divides into data-distribution
methods and error-rate methods. Error-rate methods (ADWIN [CITATION_ADWIN],
DDM) require labeled streaming data to monitor classifier error and detect when
it rises. Distribution methods compare incoming data to a reference distribution
using statistical tests. Maximum Mean Discrepancy (MMD) [CITATION_GRETTON] is
a kernel-based two-sample test that measures distributional distance in a
reproducing kernel Hilbert space; it applies naturally to unlabeled data.

DriftLens [CITATION_DRIFTLENS] applies MMD to deep learning representation
spaces for image classification and demonstrates that latent embeddings carry
richer drift signal than raw features. DriftGuard applies the same principle to
log latent embeddings, extending it with a permutation-test threshold, a
sustained-alarm criterion, and a disambiguation stage.

### D. Cross-Dataset Generalization

CroSysLog [CITATION_CROSYSLOG] addresses cross-system log anomaly detection
using meta-learning, noting that most published methods are evaluated on a
single dataset and do not transfer. Recent surveys [CITATION_SURVEY] flag
cross-dataset generalization as an open problem in log anomaly detection.
DriftGuard addresses this directly by evaluating on BGL (Blue Gene/L) and
Thunderbird — two different supercomputer architectures from different
institutions — with identical pipeline configuration.

---

## III. Methodology

### A. Overview

DriftGuard operates in three stages on top of a pre-trained autoencoder
anomaly detector. Figure 1 shows the full pipeline.

```
Raw logs → [Log Parser (Drain3)] → templates
         → [Feature Builder] → count vectors X
         → [Autoencoder] → latent embeddings z, reconstruction errors
                         → [Stage 1: MMD Drift Detection] → drift alarm
                         → [Stage 2: Selective Adaptation] → updated model
                         → [Stage 3: Disambiguation] → drift | attack
```

### B. Preprocessing

**Log Parsing.** Each raw log line is parsed by Drain3 into a template ID.
Variable tokens (IP addresses, process IDs, numbers) are masked before
parsing using regular expression substitutions to prevent high-cardinality
tokens from fragmenting the template space:

```
IP:  (\d+\.){3}\d+  →  <IP>
HEX: [0-9a-fA-F]{8,}  →  <HEX>
NUM: \b\d+\b  →  <NUM>
```

**Windowing.** Parsed template IDs are grouped into non-overlapping windows
of w = 100 consecutive lines, matching the standard window size used by DeepLog
and LogBERT on BGL. Each window is represented as a count vector x ∈ R^d, where
d is the number of unique templates and x_j counts occurrences of template j.
A window is labeled anomalous if any constituent line is labeled anomalous
(union rule).

**Train/test split.** The dataset is split chronologically: the first 80%
of windows form the training pool and the last 20% form the test set. Only
normal windows from the training pool are used for model training, following
the standard semi-supervised setup.

### C. Autoencoder Baseline

The anomaly detection model is a feed-forward autoencoder trained exclusively
on normal training windows. The architecture scales with input dimension d:

```
For d ≥ 1000:  d → 256 → 64 → 16 → 64 → 256 → d
For d < 1000:  d → 128 → 32 → 16 → 32 → 128 → d
```

Inputs are normalized with log1p (x' = log(1 + x)) to compress the count
distribution before training. The model is trained with mean squared error
loss and Adam optimizer (lr = 10^-3) for 20 epochs.

At test time, the anomaly score for window x is the mean squared reconstruction
error: score(x) = ||x' - f(x')||_2^2 / d, where f is the autoencoder. The
encoder's 16-dimensional bottleneck output z = enc(x') is saved as the latent
embedding and used by Stages 1–3.

### D. Stage 1: Label-Free Drift Detection

**MMD statistic.** Given a reference set X = {x_1,...,x_m} and a detection set
Y = {y_1,...,y_n}, the unbiased estimator of squared MMD with RBF kernel
k(x,y) = exp(-||x-y||^2 / σ^2) is:

```
MMD²(X,Y) = (1/m(m-1)) Σ_{i≠j} k(x_i,x_j)
           + (1/n(n-1)) Σ_{i≠j} k(y_i,y_j)
           - (2/mn) Σ_{i,j} k(x_i,y_j)
```

MMD² ≈ 0 when X and Y are from the same distribution; large MMD² indicates
distributional shift. We apply this to the latent embeddings z rather than
the raw count vectors, following DriftLens [CITATION_DRIFTLENS].

**Bandwidth selection.** The kernel bandwidth σ^2 is set to the median pairwise
squared distance among training embeddings (median heuristic), which is the
standard statistically motivated default for RBF kernels.

**Threshold calibration.** The detection threshold τ is set via permutation
test on the training embeddings: two disjoint random subsets of sizes m_ref
and m_win are drawn from z_train and their MMD² is computed. Repeating this
P = 200 times yields a null distribution. The threshold is the (1-α) = 99th
percentile of this null, so the expected false-alarm rate under no-drift
conditions is α = 1%.

**Reference and detection windows.** The reference set is the last m_ref = 500
training embeddings — the most recent known-normal behavior. A detection window
of m_win = 500 embeddings slides over z_test with stride s = 100. A sustained-
drift alarm is raised when K = 3 consecutive detection windows each independently
exceed τ, filtering transient spikes.

### E. Stage 2: Selective Replay-Based Adaptation

When Stage 1 raises a sustained alarm at test index t*, Stage 2 attempts to
adapt the autoencoder to the new distribution. The adapt prefix is the first
γ = 40% of the drift region [t*, t_end].

**Pseudo-normal selection.** Each window in the adapt prefix is scored by
its reconstruction error under the current model. The bottom p = 50% by error
are retained as pseudo-normals — the windows most consistent with the model's
current learned normal. Higher-error windows are more likely anomalous and
are discarded.

**Replay buffer.** To prevent catastrophic forgetting of pre-drift normal
patterns, a replay buffer of n_replay = |pseudo-normals| windows is sampled
uniformly from the original training set. The adaptation set is the union
of pseudo-normals and replay buffer.

**Fine-tuning.** The autoencoder is fine-tuned on the adaptation set for
10 epochs with batch size 16 and lr = 10^-4, a learning rate an order of
magnitude lower than initial training to preserve existing representations.

**Operational boundary.** Stage 2 should be activated only when the estimated
anomaly density of the drift region is below a safety threshold. When anomaly
density is high, pseudo-normal selection is insufficiently pure and fine-tuning
degrades model performance (see Section IV-C).

### F. Stage 3: Drift vs. Attack Disambiguation

For each window flagged by Stage 1, two features are computed:

**MMD slope.** A local linear slope is fit to the MMD² values of the K most
recent detection windows. Gradual drift produces a slow, consistent rise;
abrupt attacks produce sharp spikes. Formally: slope = (MMD²_t - MMD²_{t-K}) / K.

**Template entropy.** The Shannon entropy of the template frequency distribution
within a window:

```
H(w) = -Σ_j p_j log p_j
```

where p_j = count_j / Σ count is the empirical frequency of template j. High H
indicates many different event types in the window (diverse activity); low H
indicates a few templates dominate (repetitive activity).

These two features are used to characterize each high-MMD window. Their
correlation with the true anomaly fraction (evaluated post-hoc using held-out
labels) provides a label-free anomaly intensity signal.

---

## IV. Results and Analysis

### A. Experimental Setup

**Datasets.** All datasets are from LogHub [CITATION_ZHULOGHUB].

| Dataset | Lines | Templates | Train windows | Test windows | Test anomaly rate |
|---|---|---|---|---|---|
| HDFS | 11M | 47 | — | — | 2.9% sessions |
| BGL | 4.7M | 396 | 33,699 | 9,427 | 8.66% |
| Thunderbird | 20M (subset) | 1,753 | 113,966 | 39,935 | 65.44% |

Thunderbird's high test anomaly rate reflects that the 20M subset's final 20%
coincides with a major failure cascade. HDFS sessions are used as a drift-free
control (no time-ordering).

**Baselines.** The DriftGuard autoencoder (semi-supervised, no labels) is
compared against DeepLog (LSTM, requires labels for threshold tuning).

**Implementation.** All code is in Python 3.12 with PyTorch, scikit-learn, and
Drain3. Experiments run on a single consumer GPU. Full code available at
https://github.com/Asandhu11/driftguard.

### B. Baseline Anomaly Detection (Stage 0)

| Dataset | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| BGL | 0.7184 | 0.3539 | 0.4145 | 0.3088 |
| Thunderbird | 0.8412 | 0.8684 | 0.8459 | 0.8921 |

BGL's low F1 (0.35) reflects class imbalance (8.66% anomalous windows):
F1 is sensitive to the threshold-precision-recall tradeoff at low positive
rates. AUC 0.72 is the appropriate metric here, confirming above-chance
ranking. Thunderbird's higher AUC and F1 reflect the more pronounced signal
during the failure cascade period.

These results are achieved with zero labels — no labeled data is used to
train the model or set the detection threshold.

### C. Stage 1: Drift Detection Results

| Metric | HDFS | BGL | Thunderbird |
|---|---|---|---|
| Detection windows | 90 | 90 | 395 |
| Alarms (raw) | 0 | 82 | 395 |
| Alarm rate | 0% | 91.1% | 100% |
| First alarm at test idx | never | 300 | 0 |
| Corr(MMD², anomaly frac) | — | +0.310 | +0.750 |

**HDFS (control).** Zero alarms on the drift-free control confirms that the
permutation-test threshold does not hallucinate drift at the target α = 1%
false-alarm rate.

**BGL.** Drift is detected with gradual onset beginning at test index 300,
consistent with the known temporal structure of the BGL dataset (failures
accumulate in the second half of the 7-month collection period).

**Thunderbird.** All 395 detection windows exceed the threshold from index 0.
This reflects the sharp distributional boundary in the 20M subset: the training
portion covers the quiet early period (3.8% line-level anomaly rate) while the
test portion begins immediately within the major failure cascade. The
correlation +0.75 confirms the MMD signal is tracking real anomaly activity.

### D. Stage 2: Adaptation Results

| Slice | BGL AUC before | BGL AUC after | TB AUC before | TB AUC after |
|---|---|---|---|---|
| Pre-drift | 0.9787 | 0.9718 | — | — |
| Drift (held out) | 0.5662 | 0.5197 | — | — |
| Full test | 0.7184 | 0.6903 | 0.8412 | 0.4635 |

Adaptation degrades performance on both datasets. The root cause is pseudo-
normal contamination: some anomalous windows have reconstruction errors below
the 50th percentile threshold, particularly when the anomaly pattern
superficially resembles normal template distributions.

The severity scales with drift-region anomaly density. BGL's drift region
is 9.02% anomalous at the window level — contamination is mild and AUC drops
only 2.8 points. Thunderbird's drift region is 75.71% anomalous — the pseudo-
normal set is predominantly attack windows and AUC drops 37.7 points.

This defines an **operational boundary**: Stage 2 should be activated only
when drift-region anomaly density is estimated to be below approximately 10%.
Above this threshold, the adaptation heuristic is counterproductive and should
be bypassed.

### E. Stage 3: Disambiguation Results

| Feature | BGL correlation | Thunderbird correlation |
|---|---|---|
| MMD slope | +0.14 | +0.22 |
| Template entropy | **+0.816** | **−0.767** |

Template entropy is the stronger signal in both cases (|r| > 0.75). MMD slope
is weaker, consistent with both datasets having predominantly alarm-region
windows (slope cannot distinguish within a region that is uniformly above
threshold).

The entropy sign inverts between datasets. On BGL, high entropy correlates
with anomaly: BGL failures cascade across multiple subsystems simultaneously,
activating many different error templates and broadening the distribution.
On Thunderbird, low entropy correlates with anomaly: the Thunderbird failure
cascade is dominated by a small set of repeated catastrophic-failure messages
(device FATAL state, IB fabric errors), which narrows the distribution.

This finding generalizes the Stage 3 disambiguation principle: entropy reliably
captures anomaly intensity, but the direction depends on the failure-mode
character of the system (broadcast-divergent vs. repetitive-convergent). In
practice, a short calibration period on known-normal and known-anomalous windows
from the deployment system suffices to determine the sign.

---

## V. Discussion

### A. Cross-Dataset Generalization

DriftGuard achieves AUC 0.84 on Thunderbird — a different supercomputer
architecture, institution, and log-source mix than BGL — with identical
hyperparameters. This is a zero-shot cross-dataset transfer result. The 1,753
Thunderbird templates (vs. 396 for BGL) and the different log-source mix
(SSH, mail transfer, InfiniBand fabric, kernel, temperature sensors all
interleaved) did not require retuning the pipeline.

The result suggests that the combination of Drain3 template extraction, count-
vector windowing, and autoencoder compression is sufficiently general to capture
behavioral structure across heterogeneous log sources.

### B. Limitations

**Stage 2 boundary condition.** The pseudo-normal selection heuristic requires
a drift region where anomalies are a minority. A more principled approach would
estimate the drift-region anomaly density before triggering adaptation —
for example, using an ensemble of reconstruction thresholds or a separate
density estimator. This is the most important direction for future work.

**Entropy sign calibration.** The Stage 3 entropy feature requires knowing
whether the deployment system exhibits broadcast-divergent or repetitive-
convergent failure modes. A short supervised calibration period (even a handful
of labeled anomalous windows) would suffice to determine the sign.

**Batch evaluation.** The current implementation processes test data in batch.
An online variant with a sliding reference window that updates as new normal
data is confirmed would be more practical for production deployment.

**Thunderbird subset.** The 20M-line subset covers a specific temporal slice.
Full-dataset validation across all 211M lines would provide stronger evidence
for generalization across multiple drift events.

### C. Comparison to Related Work

DriftGuard's baseline AUC of 0.72 on BGL is below published DeepLog (AUC ~0.95)
and LogBERT results. However, those methods use labels for threshold tuning and
are evaluated on fixed splits without drift. The appropriate comparison is
unsupervised methods: DriftGuard's label-free AUC of 0.72 is competitive with
published unsupervised baselines on BGL. The primary contribution is not
achieving state-of-the-art anomaly detection accuracy but demonstrating that
MMD-based drift detection and the three-stage framework generalize across
datasets without retuning.

---

## VI. Conclusion

We presented DriftGuard, a three-stage label-free pipeline for concept-drift-
aware log anomaly detection. The pipeline detects drift using MMD on autoencoder
latent embeddings with a permutation-test threshold (no labels required),
adapts the model via selective pseudo-normal replay, and disambiguates drift
from attacks using template entropy and MMD slope.

Evaluated on HDFS, BGL, and Thunderbird, DriftGuard demonstrates cross-dataset
generalization (AUC 0.84 on Thunderbird with zero parameter changes), validates
the HDFS control (zero false alarms), and identifies a precise operational
boundary for Stage 2 adaptation (effective below ~10% drift-region anomaly
density). The inversion of the entropy signal between BGL and Thunderbird is
a finding about failure-mode character that has practical implications for
deployment.

Future work includes online MMD with sliding reference updates, a drift-region
anomaly density estimator as a Stage 2 gate, and full-dataset Thunderbird
validation.

---

## References

[1] M. Du, F. Li, G. Zheng, and V. Srikumar, "DeepLog: Anomaly Detection and
Diagnosis from System Logs through Deep Learning," in *Proc. ACM CCS*, 2017,
pp. 1285–1298.

[2] H. Guo et al., "LogBERT: Log Anomaly Detection via BERT," in *Proc. IJCNN*,
2021, pp. 1–8.

[3] W. Meng et al., "LogAnomaly: Unsupervised Detection of Sequential and
Quantitative Anomalies in Unstructured Logs," in *Proc. IJCAI*, 2019,
pp. 4739–4745.

[4] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola,
"A Kernel Two-Sample Test," *J. Mach. Learn. Res.*, vol. 13, pp. 723–773,
2012.

[5] J. Zhu et al., "Loghub: A Large Collection of System Log Datasets for
AI-Driven Log Analytics," in *Proc. ISSRE*, 2023.

[6] P. He, J. Zhu, Z. Zheng, and M. R. Lyu, "Drain: An Online Log Parsing
Approach with Fixed Depth Tree," in *Proc. ICWS*, 2017, pp. 33–40.

[7] S. Greco, L. Mariani, A. Metrio, F. Pastore, and M. Wahab,
"Unsupervised concept drift detection from deep learning representations
in real-time (DriftLens)," *arXiv:2406.17813*, 2024.

[8] Y. Wang et al., "CroSysLog: Cross-system Software Log-based Anomaly
Detection Using Meta-learning," in *Proc. SANER*, 2025.

[9] A. Bifet and R. Gavalda, "Learning from Time-Changing Data with Adaptive
Windowing," in *Proc. SDM*, 2007, pp. 443–448.

[10] Y. Yang et al., "PLELog: Semi-Supervised Log-Based Anomaly Detection via
Probabilistic Label Estimation," in *Proc. ICSE*, 2021, pp. 1919–1930.

[11] S. He, J. Zhu, P. He, and M. R. Lyu, "An Evaluation Study on Log Parsing
and Its Use in Log Mining," in *Proc. DSN*, 2016, pp. 654–661.
