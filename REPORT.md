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

### 3.4 Stage 2: Selective Replay-Based Adaptation
*[draft pending]*

### 3.5 Stage 3: Drift vs. Attack Disambiguation
*[draft pending]*

---

## 4. Experimental Setup

### 4.1 Datasets
*[draft pending]*

### 4.2 Log Parsing and Feature Construction
*[draft pending]*

### 4.3 Train/Test Splits
*[draft pending]*

### 4.4 Baselines and Metrics
*[draft pending]*

---

## 5. Results

### 5.1 Static Baselines Establish the Gap
*[draft pending]*

### 5.2 Stage 1 — Label-Free Drift Detection
*[draft pending]*

### 5.3 Stage 2 — Adaptation Tradeoff
*[draft pending]*

### 5.4 Stage 3 — Drift vs. Attack Disambiguation
*[draft pending]*

---

## 6. Discussion

*[draft pending]*

---

## 7. Conclusion

*[draft pending]*

---

## References

*[Will be compiled at the end.]*