# DriftGuard: Concept-Drift-Aware Unsupervised Anomaly Detection in System Logs for Security

**Author:** [Your Name]
**Supervisor:** Prof. Yinning Zhang
**Summer 2026 Research Internship — University of West Georgia**

---

## Abstract

*[To be written last, after all other sections are done.]*

---

## 1. Introduction

## 1. Introduction

System logs are a primary data source for detecting security incidents — intrusions, insider threats, and service failures — in modern IT and cloud infrastructures. Machine-learning-based log anomaly detection (log AD) has become the dominant approach, with deep models such as DeepLog (Du et al., 2017), LogAnomaly (Meng et al., 2019), and LogBERT (Guo et al., 2021) establishing strong baselines on public benchmarks.

These methods share an assumption that rarely holds in deployment: that the distribution of normal log behavior is stationary. In practice, software is patched, services are added or removed, user populations shift, and log templates evolve. This phenomenon — known as **concept drift** — silently degrades detection accuracy. Drifted normal behavior is misclassified as anomalous, producing false positives, while genuine attacks blend into the new "normal" and are missed.

Two further problems compound this. First, although many methods describe themselves as unsupervised, in practice they require labeled anomalies to tune detection thresholds — labels that are scarce, expensive, and slow to obtain in security settings. Second, when a system does detect a deviation, current methods cannot distinguish whether it represents legitimate drift (the system has evolved and the model should adapt) or a genuine security event (the system is under attack and an alert should be raised). These responses are opposite: adapt versus alert. Conflating them leads either to alert fatigue or to attackers being absorbed into the model's notion of normal behavior.

This report presents **DriftGuard**, a three-stage pipeline that addresses these three problems together for unsupervised log anomaly detection. The pipeline is built around a standard count-vector autoencoder and adds:

- **Stage 1 — Label-free drift detection.** Maximum Mean Discrepancy (MMD) between current and reference autoencoder latent embeddings, with a detection threshold calibrated via permutation test on training data — requiring no labeled anomalies.

- **Stage 2 — Selective replay-based adaptation.** When drift is detected, the autoencoder is fine-tuned on a mixture of low-reconstruction-error windows from the drifted region and a replay buffer drawn from the original training set, preserving the prior notion of normal while adapting to the new one.

- **Stage 3 — Drift vs. attack disambiguation.** Per-window features (MMD slope and template entropy) are computed for every high-MMD event and used to separate gradual drift from abrupt attacks, gating which response — adapt or alert — is appropriate.

The contribution is not any single component — MMD, autoencoders, and replay buffers all exist in prior work — but their integration into a single label-free pipeline tailored to security log streams. To the best of our knowledge, no published method addresses all three problems together for this setting.

DriftGuard is evaluated on two standard benchmarks from the LogHub repository (Zhu et al., 2023): **HDFS_v1**, a saturated benchmark used as a non-time-ordered control, and **BGL**, time-ordered supercomputer logs from a seven-month deployment, which serves as the main testbed for the drift experiments. The evaluation surfaces three findings that shape the discussion. First, label-free drift detection is feasible: MMD on autoencoder embeddings cleanly separates drift-bearing windows from stationary periods on BGL, while producing zero false alarms on the HDFS control. Second, the dominant lever for adaptation is not replay buffer size but the selection of "drifted normal" candidates — and aggressive candidate selection improves drift-region performance only by absorbing labeled anomalies as normal, empirically motivating the disambiguation stage. Third, the localization signal proposed in the project's initial design (template entropy) correlates strongly with attack content on BGL (+0.82), but with the opposite sign from the original hypothesis — BGL's cascading supercomputer failures produce broad rather than narrow template usage. The disambiguator works; the localization rule is dataset-dependent.

The remainder of this report is organized as follows. Section 2 surveys related work and identifies the research gap. Section 3 presents the DriftGuard methodology. Section 4 describes the experimental setup. Section 5 reports results for all three stages. Section 6 discusses limitations and implications, and Section 7 concludes.

---

## 2. Related Work

### 2.1 Static Log Anomaly Detection

Static log anomaly detection methods learn a model of normal log behavior from historical data and flag deviations at inference time. Three approaches dominate the recent literature.

**Sequence-based deep models** treat each session or window as a sequence of template identifiers and learn a next-template prediction model on normal data, flagging windows whose actual continuations diverge from the predicted distribution. DeepLog (Du et al., 2017) uses an LSTM; LogAnomaly (Meng et al., 2019) augments this with semantic template embeddings; LogBERT (Guo et al., 2021) uses a self-supervised BERT-style objective. More recent work adopts transformers (LogFormer, 2024) and large language models (LogGPT, 2023; LogFiT, 2024) for the same task.

**Reconstruction-based models** use autoencoders, variational autoencoders, or other generative models to score windows by reconstruction error: anomalies, having not been seen during training, reconstruct poorly. These are conceptually simpler and often competitive on benchmarks dominated by rare-template anomalies.

**Classical statistical baselines** — PCA, isolation forests, and one-class SVMs on count-vector or TF-IDF features — remain widely cited and sometimes match deep models on saturated benchmarks (He et al., 2016).

A limitation common to all of these is the assumption that the distribution of normal logs is stationary. None explicitly handles the concept drift that occurs in long-running deployments. Recent surveys (Landauer et al., 2023; AIOps for log anomaly detection in the era of LLMs, 2025) flag this gap repeatedly.

### 2.2 Concept Drift Detection

Concept drift detection in machine learning streams is a mature research area, with a comprehensive recent survey by Hinder, Vaquet, and Hammer (2024). Methods broadly fall into two families: **supervised detectors** that monitor prediction performance over time (and require ongoing access to labels), and **unsupervised detectors** that monitor the input distribution directly.

Within the unsupervised family, two recent methods are most relevant. **DriftLens** (Greco et al., 2024) detects drift in the latent representations of deep classifiers using distribution distance measures and is explicitly designed to be label-free; however, it is evaluated on image, text, and audio classifiers, not on log anomaly detection, and includes no adaptation step. **VAE4AS** (Li et al., 2024) combines a variational autoencoder with dual drift detection (statistical plus distance-based) for anomalous-sequence detection and includes an incremental learning component; however, it is evaluated on generic time series rather than system logs, and treats every detected drift event as an adaptation trigger — it does not separate drift from attacks.

**Maximum Mean Discrepancy (MMD)** is the kernel-based two-sample test underlying both DriftLens and many subsequent drift detectors (Gretton et al., 2012). Given two sets of samples, MMD with an RBF kernel estimates whether they come from the same distribution. Its standard practical choice — the median-heuristic bandwidth — and its unbiased estimator are well-established; this work uses both.

### 2.3 Cross-System and Drift-Aware Log AD

A related thread of recent work addresses cross-system generalization for log AD. ZeroLog (Wang et al., 2025), LogMoE (ASE 2025), CroSysLog (Wang et al., 2024), and MetaLog (2024) use meta-learning, mixtures of experts, and domain adaptation to transfer log AD models across systems with different log distributions. These methods address a distinct but related problem: train-test distribution mismatch where the target distribution is fixed. They do not handle a continuously evolving stream within a single system, and they do not address the question of when a detected shift represents drift versus an attack.

A small number of recent papers do touch on concept drift specifically within log anomaly detection — most notably LightLog (2022), which proposes a lightweight LSTM with brief concept-drift consideration for edge deployment. As of a literature search conducted at the start of this project (post-2022, on Google Scholar), the number of dedicated publications on concept drift in unsupervised log anomaly detection is on the order of ten to fifteen, compared to several hundred on log AD overall. This is an under-studied area.

### 2.4 Research Gap

Three problems are addressed in isolation by prior work, but no published method addresses all three together for security log streams:

1. **Label-free drift detection on logs.** DriftLens does this for generic deep classifiers; no method does it for log AD specifically.

2. **Selective adaptation to drift without forgetting.** VAE4AS does this for generic anomalous sequences; no log AD method does it with explicit replay buffering and no labeled supervision.

3. **Drift vs. attack disambiguation.** No method, log AD or otherwise, explicitly distinguishes drift from attacks so that the response can be selected accordingly.

DriftGuard's contribution is to integrate these three capabilities into a single pipeline tailored to security log streams.

---

## 3. Method

### 3.1 Pipeline Overview

DriftGuard wraps three stages around any embedding-based log anomaly detector. The base detector used in this work is a feed-forward autoencoder trained on parsed log-template count vectors. The three stages added on top operate on the detector's outputs and latent representations:

- **Stage 1 — Drift detection.** A reference window of latent embeddings is stored at training time. For each new detection window over the test stream, the squared Maximum Mean Discrepancy (MMD²) between the window's embeddings and the reference is computed. A threshold derived from a permutation test on the training embeddings (no labels) flags windows as drifted.

- **Stage 2 — Adaptation.** When drift is detected, the autoencoder is fine-tuned on a mixture of low-reconstruction-error windows from the drifted region ("drifted normal" candidates) and a replay buffer of samples from the original training set.

- **Stage 3 — Disambiguation.** Each detection window receives two additional features: the slope of MMD over a short sliding window (temporal sharpness) and the Shannon entropy of the template-count distribution within the window (template localization). Together with MMD magnitude, these features separate drift events from attack events.

The pipeline is designed to be modular: each stage has a defined input/output contract and can be evaluated independently, which is how the experimental results in Section 5 are structured.

### 3.2 Base Autoencoder

The base detector is a feed-forward autoencoder with a symmetric encoder-decoder structure. Given an input count vector of dimension *d*, the architecture is *d* → *h₁* → *h₂* → *latent* → *h₂* → *h₁* → *d*, with ReLU activations between linear layers and no activation on the final output layer. Hidden layer sizes scale with input dimension to keep the parameter count proportional to the feature dimensionality:

| Input dim *d* | (*h₁*, *h₂*, latent) |
|---|---|
| < 100 | (32, 16, 8) |
| 100 – 1000 | (128, 32, 16) |
| ≥ 1000 | (256, 64, 16) |

Inputs are normalized with log1p transformation (log(1+x) elementwise) before training. This compresses heavy-tailed template counts — common in log data, where a handful of templates dominate — without distorting zero-valued entries. The autoencoder is trained on normal sessions only using Adam (learning rate 1e-3, batch size 256) for 20 epochs, minimizing mean squared error.

At inference time, an input's anomaly score is its mean squared reconstruction error across the *d* output dimensions. Anomalous inputs, having distributions unseen during training, reconstruct poorly and receive higher scores.

### 3.3 Stage 1 — Label-Free Drift Detection

Stage 1 detects whether the distribution of incoming logs has drifted from the training distribution, **using no labeled anomalies at any point**. The construction has three components.

**Latent representations.** After training, the encoder is frozen and used to compute latent embeddings *z* ∈ ℝᵏ for every input in both the training set and the test stream. Operating in the encoder's latent space rather than the raw input space concentrates distribution differences along the directions the model considers semantically meaningful, and reduces the cost of kernel computations (latent dimension is 8 or 16 versus 47–1822 in raw counts).

**MMD with RBF kernel.** Given two sets of latent embeddings *X* = {*x₁*, …, *x_m*} and *Y* = {*y₁*, …, *y_n*}, the unbiased squared MMD with RBF kernel *k(x, y) = exp(-‖x − y‖² / σ²)* is

> MMD²(X, Y) = (1 / m(m−1)) Σᵢ≠ⱼ k(xᵢ, xⱼ) + (1 / n(n−1)) Σᵢ≠ⱼ k(yᵢ, yⱼ) − (2 / mn) Σᵢⱼ k(xᵢ, yⱼ)

MMD² is non-negative; values near zero indicate the two samples could plausibly come from the same distribution, while large values indicate distributional difference. The RBF bandwidth σ² is set by the **median heuristic** — σ² equals the median squared pairwise distance among training embeddings — which is the standard well-behaved default for this kernel.

**Threshold calibration via permutation test.** To detect drift without labels, a detection threshold is required that controls the false-alarm rate under "no drift." We construct an empirical null distribution by repeatedly drawing two disjoint random subsets from the training embeddings — one of size *m* (matching the reference) and one of size *n* (matching the detection window) — and computing MMD² between them. This null distribution captures the variability of MMD² under the null hypothesis that both samples come from the same (training) distribution. The detection threshold is the (1 − α) quantile of this null distribution, with α = 0.01 in our experiments. Under no drift, we expect at most α = 1% false alarms by construction.

**Sustained-drift criterion.** A single threshold crossing can be triggered by transient noise or anomaly-induced spikes. To filter these out, we require *K* consecutive detection windows to exceed the threshold before declaring sustained drift. We use *K* = 3 throughout this work.

**Detection.** At inference time, a reference window is fixed (in our experiments, the last *m* = 500 training embeddings — the most recent representation of normal at training time). A detection window of size *n* = 500 slides over the test embeddings with stride 100. At each step, MMD² between the current window and the reference is computed; if the value exceeds the threshold for *K* consecutive steps, sustained drift is reported. The label-free property — no anomaly labels touched in either threshold calibration or scoring — is the core contribution of Stage 1.

### 3.4 Stage 2 — Selective Replay-Based Adaptation

When Stage 1 reports sustained drift, Stage 2 fine-tunes the autoencoder so that subsequent reconstruction-error scores reflect the new normal — without forgetting the old normal. The adaptation procedure has three components.

**Candidate selection.** Within the drift region, we score each window with the *current* (un-adapted) autoencoder and keep the windows with the lowest reconstruction errors as "drifted normal" candidates. The intuition is that windows the current model already scores as relatively normal are the safest to use for fine-tuning: they reflect drift in benign system behavior rather than security events. We control this with a `candidate_frac` parameter that selects the bottom *k* fraction by reconstruction error.

**Replay buffer.** To resist catastrophic forgetting of the original normal distribution, we draw a random sample from the original training set as a replay buffer. The buffer size is controlled by a `replay_mult` parameter: replay size equals `replay_mult` × (number of drifted-normal candidates). Setting `replay_mult` to zero disables replay entirely; large values anchor the model heavily to the pre-drift normal.

**Fine-tuning.** The drifted-normal candidates and the replay buffer are concatenated into a mixed adaptation set. The autoencoder is fine-tuned on this set with Adam at a small learning rate (1e-4, one order of magnitude smaller than the original training rate of 1e-3) for a small number of epochs. The small step size and limited epoch count are deliberate: aggressive fine-tuning would absorb drift quickly but would also forget more of the pre-drift distribution. This is the canonical plasticity-stability tradeoff.

**Operating-point evaluation.** We evaluate the adapted model on three disjoint slices of the test set: a **pre-drift slice** (test windows before the drift onset, used to measure forgetting), a **drift held-out slice** (drift-region windows not used for adaptation, used to measure post-adaptation generalization), and the **full test set**. We report ROC-AUC and best-threshold F1 on each slice. A successful adaptation increases drift held-out F1 without measurably decreasing pre-drift F1.

### 3.5 Stage 3 — Drift vs. Attack Disambiguation

Stages 1 and 2 alone do not distinguish drift from attack: an anomaly-rich region of the test stream looks like drift to MMD, and Stage 2 will adapt to it — silently absorbing attacks into the model's notion of normal. Stage 3 provides the missing signal: when a detection window has a high MMD score, two additional features classify the event as drift-like or attack-like before any adaptation is performed.

**Feature 1: MMD slope.** Drift is by definition gradual — distributions shift over many windows. Attacks are sharp — a sudden burst of unusual log activity. We capture this by computing the slope of the MMD time series over a short context of *K_s* = 5 detection windows, using ordinary least squares fit to (window-index, MMD²) pairs. A small slope indicates a sustained level (drift); a steep positive slope indicates a sharp rise (attack).

**Feature 2: Template entropy.** For each detection window, we compute the Shannon entropy of the aggregated template-count distribution within that window:

> *H(window)* = −Σᵢ pᵢ log pᵢ

where *pᵢ* is the empirical probability of template *i* (count of template *i* divided by total counts in the window). The hypothesis informing the original proposal was that attacks would have *low* entropy (a narrow set of templates dominates, e.g., a single attacker hitting a single endpoint) and drift would have *high* entropy (system evolution affects many templates).

**Empirical inversion on BGL.** On the BGL dataset, this hypothesis is inverted (Section 5.4). BGL's labeled anomalies correspond to cascading supercomputer failures that involve many subsystems simultaneously — TLB errors, ECC memory faults, kernel exceptions, and network alerts often co-occur. Such cascading events produce *high* template entropy, while the smaller drift-only events involve narrower, system-internal template usage. The disambiguator still works, but the localization rule must be calibrated to the failure mode of the system being monitored. Section 5.4 quantifies this and discusses the implications for deployment.

**Disambiguation rule.** For each detection window with MMD² above the Stage 1 threshold, we treat the window's (slope, entropy) pair as a feature vector. A simple threshold rule — drift if entropy is low *and* slope is small, attack otherwise (with thresholds calibrated per-system) — gates the response. In the experiments we report the empirical relationship between (slope, entropy) and ground-truth anomaly fraction rather than fixing a specific decision rule, which keeps the disambiguator open to alternative calibrations as more data becomes available.

---

## 4. Experimental Setup

### 4.1 Datasets

Two public benchmarks from the LogHub repository (Zhu et al., 2023) are used.

**HDFS_v1** consists of approximately 11.2 million log lines from a Hadoop Distributed File System cluster, with 575,061 block-level sessions labeled normal or anomalous via a separate `anomaly_label.csv` file. The anomaly rate is 2.93% (16,838 anomalous sessions). HDFS_v1 is not time-ordered in any meaningful sense — block IDs are independent and sessions can be reordered without changing the problem — and is widely regarded in the recent literature as a saturated benchmark on which simple methods already achieve near-perfect performance. We use HDFS_v1 in this report as (a) a saturated reference to demonstrate that the base autoencoder is working correctly, and (b) a non-time-ordered control for the drift detector: under a random train/test split, MMD should detect no drift.

**BGL** consists of approximately 4.7 million log lines from the Blue Gene/L supercomputer at Lawrence Livermore National Laboratory, spanning seven months (June 2005 to April 2006). Each log line carries a real timestamp and a per-line alert label, with about 7.4% of lines labeled as alert events. BGL is genuinely time-ordered, which makes it the appropriate testbed for drift experiments: the system's failure patterns and software stack evolved over the seven-month window. We group consecutive lines into non-overlapping windows of 100 lines (following the standard DeepLog / LogBERT convention), yielding 47,135 windows with a window-level anomaly rate of 10.24% (a window is anomalous if it contains at least one anomaly line).

### 4.2 Log Parsing and Feature Construction

Raw log lines are parsed into structured templates using **Drain3** (a maintained implementation of Drain, He et al. 2017). Drain extracts log templates by replacing variable parts (e.g., timestamps, IP addresses, block IDs) with wildcards, reducing millions of unique log lines to a small set of recurring patterns. With default Drain3 settings, HDFS_v1 yields 47 unique templates and BGL yields 1,822 templates.

The 1,822-template count on BGL is higher than the manually curated count of ~376 templates reported in the LogPai analysis (Zhu et al., 2023). This is due to Drain3's default similarity threshold splitting some semantically similar messages into separate templates. Tuning `sim_th` upward from its default 0.4 to ~0.5 would reduce this count, at the cost of conflating some message variants. We note this as a refinement opportunity but use default settings throughout to keep results comparable to the LogHub reference distribution.

Each session (HDFS) or 100-line window (BGL) is converted into a **count vector** of dimension equal to the template universe (47 for HDFS, 1,822 for BGL): entry *i* counts the occurrences of template *i* in the session/window. Count vectors lose template *ordering* information, which is a known limitation discussed in Section 5.1. Inputs are log1p-normalized (Section 3.2) before being fed to the autoencoder.

### 4.3 Train/Test Splits

**HDFS** is split randomly: 80% of normal sessions go into the training pool (446,578 sessions) and the remainder, together with all anomalous sessions, into the test set (128,483 sessions; test anomaly rate 13.11%). Because HDFS has no time structure, a random split is appropriate.

**BGL uses a strict time-ordered split:** the first 80% of windows in chronological order form the training pool (37,708 windows), and the last 20% form the test set (9,427 windows). Of the 37,708 training-pool windows, the 33,699 labeled normal are kept for training; anomalous training-pool windows are discarded (semi-supervised setup — the autoencoder learns "normal" only). The test set contains 8,611 normal and 816 anomalous windows (test anomaly rate 8.66%). The time-ordered split is essential for the drift experiments: it preserves the natural distribution shift between early and late BGL.

### 4.4 Baselines and Metrics

Two baseline detectors are evaluated alongside the DriftGuard-augmented autoencoder:

- **Count-vector autoencoder (CV-AE).** The base autoencoder described in Section 3.2, evaluated without any DriftGuard stages.
- **DeepLog-style LSTM.** A from-scratch reimplementation of next-template prediction (Du et al., 2017): an embedding layer (size 32) feeds into a single-layer LSTM (hidden size 64); the last hidden state predicts the next template via a linear output layer over the vocabulary. The model is trained on normal sequences with cross-entropy loss for 5 epochs, with up to 2 million randomly sampled (context, target) pairs per dataset for tractable training time. At test time, each window is scored by the mean cross-entropy of its next-template predictions; higher mean cross-entropy indicates a less predictable (more anomalous) window. Note that this differs from the original DeepLog scoring (top-K matching); we discuss the implications in Section 5.1.

**Metrics.** We report ROC-AUC and best-threshold F1 throughout. ROC-AUC measures ranking quality and is threshold-independent. Best-threshold F1 sweeps all possible thresholds and reports the maximum F1; this gives an optimistic upper bound on the F1 a deployed system could achieve with perfect threshold tuning, but is the standard reporting convention for log AD benchmarks. For Stage 1 (drift detection), we additionally report the correlation between the MMD time series and the per-window true anomaly fraction, as a diagnostic of how much of the MMD signal is anomaly-driven versus distribution-shift-driven.

All experiments are run with random seed 42 for reproducibility.

---

## 5. Results

### 5.1 Static Baselines Establish the Gap

Before evaluating DriftGuard, we measure the two static baselines (count-vector autoencoder and DeepLog-style LSTM) on both datasets. These results establish the regime in which the drift-aware contributions of DriftGuard are needed.

**Table 1.** Static baseline performance on HDFS and BGL.

| Model | HDFS AUC | HDFS F1 | BGL AUC | BGL F1 |
|---|---|---|---|---|
| Count-vector AE | 0.9999 | 0.9973 | 0.7354 | 0.3467 |
| DeepLog-style LSTM | 0.6169 | 0.6700 | 0.6065 | 0.2822 |

The HDFS result for the count-vector autoencoder (AUC 0.9999, F1 0.9973) exceeds the published DeepLog F1 of approximately 0.96 (Du et al., 2017). This is not a methodological breakthrough; it is a property of the HDFS_v1 dataset. The reconstruction-error histogram (Figure 1a) shows essentially non-overlapping distributions for normal and anomalous sessions, with two orders of magnitude separation between their means. The cause is that HDFS anomalies almost always involve specific "rare error templates" (for example, *"writeBlock received exception"*) that never occur in normal sessions, making them trivially separable in a count-vector representation. Recent surveys (Landauer et al., 2023; AIOps for log anomaly detection in the era of LLMs, 2025) explicitly describe HDFS as saturated for this reason.

On BGL, the count-vector autoencoder drops to AUC 0.7354 / F1 0.3467 (Figure 1b). This is the realistic regime: count vectors lose the temporal *ordering* of templates, and BGL anomalies are largely order-based (legitimate-looking templates occurring in unusual sequences) rather than rare-template-based.

The DeepLog-style LSTM underperforms the count-vector autoencoder on HDFS (F1 0.67 vs. 0.997) and barely changes on BGL (F1 0.28). On HDFS, the LSTM cannot exploit rare templates as effectively as the count-vector AE because its scoring is averaged over many predictions. On BGL, the LSTM does not improve over the count-vector AE because of our scoring choice: we use mean cross-entropy across each window, while the original DeepLog uses top-K matching (a discrete count of positions where the actual next template falls outside the top-K predictions). Mean cross-entropy dilutes the signal — a single highly surprising transition gets averaged across dozens of normal ones. We did not implement top-K matching, accepting the weaker LSTM scoring in exchange for spending time on the project's actual contribution: drift-aware extensions to the autoencoder.

**Implication for DriftGuard.** The combination of (a) a strong, simple autoencoder baseline on count vectors and (b) a clear gap on the time-ordered BGL dataset is exactly the regime that motivates drift-aware methods. The remainder of Section 5 evaluates whether DriftGuard's three stages close this gap.

**Figure 1.** Reconstruction-error histograms for the count-vector autoencoder. (a) HDFS: normal and anomalous error distributions are widely separated (see `results/hdfs_recon_error_histogram.png`). (b) BGL: distributions overlap substantially (see `results/bgl_recon_error_histogram.png`).

### 5.2 Stage 1 — Label-Free Drift Detection

Stage 1 evaluates whether MMD on autoencoder latent embeddings detects distribution shift in the test stream without using any labeled anomalies.

**HDFS control.** When the random-split HDFS test set is fed through Stage 1, MMD remains below the calibrated threshold across all 1,280 detection windows (Figure 2a). The raw alarm count is **zero**, and consequently the sustained-alarm count (*K* = 3 consecutive crossings) is also zero. This is the expected null result: HDFS is not time-ordered, so a randomized 80/20 split produces train and test embeddings drawn from the same distribution. The absence of false alarms on this control validates the threshold-calibration procedure: under no drift, MMD does not spuriously fire.

**BGL.** On BGL's time-ordered test stream (Figure 2b), Stage 1 detects substantial drift. Out of 90 detection windows, 78 exceed the threshold (raw alarm rate 86.7%); applying the sustained-drift criterion of *K* = 3 consecutive crossings does not reduce this count, indicating that the alarms form long contiguous blocks rather than isolated spikes. The first drift alarm fires at test index 300; the MMD signal remains elevated throughout most of the remainder of the test stream. The correlation between MMD and the true per-window anomaly fraction is +0.42.

Two qualitative features of the BGL MMD curve are worth highlighting (visible in Figure 2b):

1. **Drift detected without anomalies.** Two MMD spikes — around indices 500–1100 and 2000–2500 — occur when the true anomaly fraction in the corresponding test windows is essentially zero. This demonstrates that MMD captures distribution shift that the anomaly labels do not, validating the label-free framing.

2. **Sustained drift after anomalies end.** A large rise in MMD occurs around index 3500 alongside a burst of labeled anomalies. The anomalies fade by index 5000–7000, but MMD remains elevated through the end of the test stream. Drift persists after the anomaly burst is over, demonstrating that drift and attacks are distinct signals on the same time series — directly motivating Stage 3.

**Figure 2.** Stage 1 drift detection. (a) HDFS control: MMD remains below threshold; zero alarms (see `results/hdfs_mmd_drift.png`). (b) BGL: MMD curve with sustained alarms, plus per-window true anomaly fraction below (see `results/bgl_mmd_drift.png`). Note that the early MMD spikes correspond to windows with zero anomalies.

### 5.3 Stage 2 — Adaptation Tradeoff

Stage 2 fine-tunes the autoencoder on a mixture of drifted-normal candidates and a replay buffer. We evaluate the adapted model on three slices defined in Section 3.4: pre-drift (test windows before the drift onset), drift held-out (drift-region windows not used for adaptation), and full test.

We initially expected adaptation to surface a clean plasticity-stability tradeoff controlled by the **replay multiplier** — the ratio of replay-buffer size to drifted-normal candidate count. We swept `replay_mult` ∈ {0, 0.5, 1, 2, 3, 5} at three fine-tuning intensities (5, 10, and 20 epochs). To our surprise, **replay-multiplier had no measurable effect on outcomes** across the entire sweep (Figure 3a). Pre-drift F1 remained at 0.81–0.83, drift held-out F1 at 0.42–0.47, and pre-drift AUC at 0.996 across all settings. At the model scale used here (~970,000 parameters for BGL) and the small fine-tuning footprint (~700 candidates, lr 1e-4, 5–20 epochs), the model simply does not move enough for replay-buffer size to matter.

This led us to identify the actual operative lever: **candidate selection**. We swept `candidate_frac` ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} with replay multiplier and epochs fixed (Figure 3b, Table 2).

**Table 2.** BGL Stage 2 results across `candidate_frac` (replay_mult=1.0, epochs=10).

| candidate_frac | # kept | contamination | pre F1 | drift F1 | drift AUC |
|---|---|---|---|---|---|
| 0.1 | 237 | 0.0% | 0.822 | 0.415 | 0.597 |
| 0.2 | 474 | 0.0% | 0.835 | 0.415 | 0.596 |
| 0.3 | 711 | 4.5% | 0.809 | 0.467 | 0.537 |
| 0.5 | 1,185 | 6.7% | 0.776 | 0.464 | 0.545 |
| 0.7 | 1,659 | 21.2% | 0.776 | **0.535** | **0.895** |
| 0.9 | 2,133 | 16.9% | 0.760 | 0.536 | 0.898 |

The pre-adaptation baseline F1 is 0.41 on the drift held-out slice. At `candidate_frac` = 0.7, adaptation lifts drift held-out F1 to 0.535 and drift AUC from 0.60 to 0.90 — a substantial ranking improvement on the drifted region. Pre-drift F1 drops from 0.82 to 0.76; pre-drift AUC remains essentially unchanged at 0.996, indicating that the underlying ranking on pre-drift normal is preserved (the F1 drop comes from the best-threshold shifting, not from genuine degradation of the model's ability to distinguish pre-drift normal from anomaly).

**The uncomfortable result.** The largest adaptation gains occur at `candidate_frac` ≥ 0.7, where the contamination column shows that 21% of the candidates the model is fine-tuning on are *actually labeled anomalies*. Adaptation does not so much "learn the new normal" as **absorb anomalies into the model's notion of normal**. From an evaluation standpoint this looks like an improvement; from a deployed-system standpoint it is exactly the failure mode that motivates Stage 3. Without gating, Stage 2 will silently adapt away the very alerts the system exists to raise.

This is empirically the strongest single finding of this report: drift-aware adaptation, applied naively, is dangerous. The gating provided by Stage 3 is not optional refinement — it is necessary for safe deployment.

**Figure 3.** Stage 2 sweeps. (a) `replay_mult` sweep at 10 epochs: curves are flat (see `results/bgl_stage2_sweep.png`). (b) `candidate_frac` sweep at fixed replay and epochs: drift F1 rises with candidate fraction but tracks contamination (see `results/bgl_stage2_candsweep.png`).

### 5.4 Stage 3 — Drift vs. Attack Disambiguation

Stage 3 computes two per-window features — MMD slope and template entropy — for every detection window and asks whether they can be used to separate drift-like events from attack-like events without using labels.

On the 78 high-MMD windows in the BGL test stream, the correlation between **template entropy** and the true per-window anomaly fraction is **+0.82** — a strong signal. The correlation between **MMD slope** and the anomaly fraction is **+0.245** — a weaker but consistent signal in the expected direction (steeper rises track attacks). Together these features carry substantial discriminative information.

Figure 4 (the bottom-right panel of `results/bgl_disambiguate.png`) plots each high-MMD window in (slope, entropy) space, colored by its true anomaly fraction. Two distinct clusters are visually evident:

1. **Low-entropy / low-slope cluster** (entropy < 1.0, slope near zero): dominantly blue (low anomaly fraction). These are pure-drift events — the system's template usage shifted in narrow patterns without involving labeled failures.

2. **High-entropy / positive-slope cluster** (entropy > 2.0, slope > 0.05): dominantly red (high anomaly fraction). These are attack events — cascading supercomputer failures that involve many subsystems simultaneously and produce sharp MMD rises.

**The inverted hypothesis.** The original proposal predicted attacks would have *low* template entropy (a single attacker concentrating on a narrow endpoint). On BGL the relationship is inverted: attacks have *high* entropy. The reason is BGL-specific — its labeled anomalies are not external intrusions but internal cascading failures, where one subsystem failure triggers logging in many others. CPU TLB errors, ECC memory faults, kernel exceptions, and network alerts often co-occur, so an attack window contains many templates.

This is not a refutation of the disambiguator. The (slope, entropy) feature pair clearly separates drift from attack on BGL with a +0.82 entropy correlation — the disambiguator works. What is dataset-dependent is the *sign* of the entropy axis: in deployments where attacks correspond to narrow concentrated activity (e.g., a single endpoint being probed), the proposal's original direction would hold. The implication for deployment is that the threshold rule should be calibrated per-system using a small labeled calibration set drawn from the early operating period, before the disambiguator is used to gate Stage 2.

**Figure 4.** Stage 3 disambiguator on BGL high-MMD windows. The (slope, entropy) scatter shows clean separation between pure-drift events (low entropy, low slope) and attack events (high entropy, positive slope). Per-window features are in `results/bgl_disambiguate.csv`; the four-panel diagnostic plot is in `results/bgl_disambiguate.png`.

---

## 6. Discussion

*[draft pending]*

---

## 7. Conclusion

*[draft pending]*

---

## References

*[Will be compiled at the end.]*