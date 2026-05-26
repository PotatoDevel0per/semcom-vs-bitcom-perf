# Progress Report
## Performance Comparison of Semantic Communication vs. Raw-Bit Transmission

**14897 Topics in Mobile Computing — Mini-Project Progress Report**
Student: Jihyeok Lee (20266243) | Dept. of Computer Science and Engineering (M.S. 1st Semester)
Date: April 8, 2026

---

## Abstract

This progress report summarizes the work completed to date on a performance comparison between semantic communication and raw-bit (JPEG-compressed) transmission for task-oriented mobile computing. We established a complete simulation pipeline using CIFAR-10 image classification as the downstream task. A ResNet-18 classifier, serving as the baseline, achieved  a test accuracy of 93.51 %. Semantic encoders with latent dimensions of 64, 128, and 256 were trained end-to-end and evaluated under AWGN and packet-loss channel conditions. Initial results show that the semantic approach (dim=64, 64 bytes) achieves 91.94% accuracy at SNR=10 dB — using only 2.1% of the bandwidth required by the lossless baseline — while outperforming JPEG compression by a factor of approximately 15× in accuracy per bit.

---

## 1. Introduction

The proliferation of AI-driven mobile applications demands efficient data transmission for inference tasks over bandwidth-constrained wireless channels. Conventional raw-bit transmission, even with source coding such as JPEG, transmits pixel-level information regardless of the downstream task, leading to poor utilization of the limited channel resources. Semantic communication addresses this issue by extracting and transmitting only task-relevant features, thereby reducing the payload while preserving task performance [1][2].

This project quantitatively compares semantic communication and JPEG-based raw-bit transmission under realistic wireless channel conditions, focusing on three metrics: Top-1 classification accuracy, accuracy per bit (ApB), and robustness to channel degradation (SNR variation and packet loss).

---

## 2. Problem Statement

Given a bandwidth-constrained wireless channel, the transmitter must send a representation of an image $x$ such that the receiver can correctly classify it. The key tension is between the transmission size (bytes) and task accuracy. Formally, for a transmission budget $B$ (bytes), we compare

- **Baseline:** JPEG compression at quality $q$, where image $x$ is compressed to approximately $B$ bytes, transmitted, decompressed and fed to a classifier.
- **Semantic:** A learned encoder $f_\phi(x) \in \mathbb{R}^d$ extracts a $d$-dimensional latent vector, quantized to $d$ bytes via 8-bit scalar quantization, transmitted through a noisy channel and classified by a lightweight head $g_\psi$.

The channel is modeled as an AWGN with varying SNR, Rayleigh fading, or block-level packet loss.

---

## 3. Work Completed

### 3.1 Simulation Environment

The full experimental pipeline was implemented in Python/PyTorch and version-controlled on GitHub. The system runs on an Apple MPS (M-series GPU). Key components:

- **Dataset:** CIFAR-10 (50,000 train / 10,000 test, 32×32 RGB)
- **Channel simulators:** AWGN, Rayleigh fading, block-level packet loss
- **Quantizer:** 8-bit uniform scalar quantization
- **Baseline:** JPEG compression via PIL at quality levels q ∈ {1, 3, 5, 10, 20, 40, 60, 80, 95}

### 3.2 Baseline: ResNet-18 Classifier

A ResNet-18 classifier was adapted for CIFAR-10 (3×3 first conv, stride 1, no max-pooling) and trained for 30 epochs using SGD with cosine annealing (lr=0.1, weight decay=5×10⁻⁴, momentum=0.9). The model achieved a test accuracy of 93.51 %, which served as the upper-bound reference for lossless transmission.

### 3.3 Semantic Encoder Training

Three SemanticEncoder models (latent dim ∈ {64, 128, 256}) were trained end-to-end using the same ResNet-18 backbone with a linear projection head paired with a two-layer MLP classifier head. Training used Adam (lr=1×10⁻³, cosine annealing, 30 epochs). Results:

| Latent Dim | Transmission Size | Best Test Accuracy |
|:----------:|:-----------------:|:------------------:|
| 64         | 64 B              | 92.52%             |
| 128        | 128 B             | 92.25%             |
| 256        | 256 B             | 92.42%             |

### 3.4 Baseline Evaluation (JPEG)

JPEG compression was evaluated at nine different quality levels. The classifier (ResNet-18, §3.2) was applied to decompressed images without retraining.

| Method       | Avg. Bytes | Top-1 Acc | Acc/Bit (×10⁻⁴) |
|:-------------|:----------:|:---------:|:----------------:|
| Raw (lossless) | 3,072 B  | 93.51%    | 0.38             |
| JPEG q=10    | 699.5 B    | 42.47%    | 0.76             |
| JPEG q=40    | 801.3 B    | 72.63%    | 1.13             |
| JPEG q=60    | 853.8 B    | 78.83%    | 1.15             |
| JPEG q=80    | 960.6 B    | 85.36%    | 1.11             |
| JPEG q=95    | 1,300.8 B  | 91.70%    | 0.88             |

JPEG reaches its peak accuracy per bit at q=60 (1.15×10⁻⁴) but requires over 850 bytes for ~79% accuracy.

### 3.5 Semantic Evaluation

Semantic encoders were evaluated under AWGN (SNR ∈ {0, 5, 10, 20} dB) and packet loss (rate ∈ {1%, 5%, 10%}).

**Table 3. Semantic accuracy under AWGN (CIFAR-10 test set)**

| Latent Dim | SNR=0 dB | SNR=5 dB | SNR=10 dB | SNR=20 dB |
|:----------:|:--------:|:--------:|:---------:|:---------:|
| 64         | 88.67%   | 91.32%   | 91.94%    | 92.52%    |
| 128        | 90.49%   | 91.92%   | 92.14%    | 92.40%    |
| 256        | 91.43%   | 92.16%   | 92.16%    | 92.37%    |

**Table 4. Semantic accuracy under packet loss**

| Latent Dim | Loss=1% | Loss=5% | Loss=10% |
|:----------:|:-------:|:-------:|:--------:|
| 64         | 92.46%  | 92.24%  | 92.32%   |
| 128        | 92.21%  | 92.32%  | 92.18%   |
| 256        | 92.40%  | 92.41%  | 92.45%   |

**Table 5. Communication efficiency at SNR=10 dB**

| Method           | Bytes | Top-1 Acc | Acc/Bit (×10⁻⁴) |
|:-----------------|:-----:|:---------:|:----------------:|
| JPEG q=60 (best ApB) | 853.8 | 78.83% | 1.15          |
| JPEG q=95        | 1,300.8 | 91.70%  | 0.88             |
| **Semantic dim=64**  | **64** | **91.94%** | **17.96**    |
| Semantic dim=128 | 128   | 92.14%    | 8.99             |
| Semantic dim=256 | 256   | 92.16%    | 4.50             |

---

## 4. Preliminary Findings

**(1) Dramatic bandwidth reduction:** Semantic dim=64 achieves 91.94% accuracy using only **64 bytes** — 2.1% of the 3,072-byte lossless baseline — while JPEG requires 1,301 bytes to reach comparable accuracy (91.70%).

**(2) Communication efficiency:** Semantic dim=64 achieves Accuracy per Bit of 17.96×10⁻⁴, approximately **15.6× higher** than the best JPEG configuration (q=60, 1.15×10⁻⁴).

**(3) AWGN robustness:** Even at SNR=0 dB, semantic dim=256 maintains 91.43% accuracy. The dim=64 model shows greater sensitivity to low SNR (88.67% at 0 dB) owing to its compressed representation.

**(4) Packet loss robustness:** All semantic models are remarkably robust to packet loss. Dim=256 accuracy fluctuates by less than 0.05% across 1%–10% loss rates, suggesting that the latent representation distributes task-relevant information redundantly across the dimensions.

---

## 5. Updated Project Schedule (Gantt Chart)

| Period | Activity | Milestone |
|--------|----------|-----------|
| 2026-03-13 ~ 03-17 | Environment setup, baseline training | — |
| **2026-03-18** | **Proposal submission** | **Proposal ✓** |
| 2026-03-19 ~ 04-07 | Semantic encoder development, initial experiments | — |
| **2026-04-08** | **Progress Report submission** | **Progress Report ✓** |
| 2026-04-09 ~ 04-20 | Extended experiments: Rayleigh fading, Tiny-ImageNet | — |
| 2026-04-21 ~ 05-10 | Statistical repetition (5 runs), result consolidation | — |
| 2026-05-11 ~ 05-20 | Final report writing, slide preparation | — |
| **2026-05-27** | **Presentation** | **Slides + Demo** |
| **2026-06-03** | **Final report submission** | **IEEE format** |

**Changes from original plan:** No major deviations. The full evaluation of Rayleigh fading was deferred to the next phase (April 9–20) to prioritize the submission of the Progress Report.

---

## 6. Remaining Work

- **Extended channel evaluation:** Full Rayleigh fading sweep across all latent dims
- **Statistical rigor:** Repeat all experiments 5 times (seed sweep) and report mean ± std
- **Tiny-ImageNet:** Extend dataset if time permits (originally planned as optional)
- **Vector quantization:** Compare K-means VQ against scalar quantization
- **Final report:** IEEE conference format, 10+ references

---

## References

[1] Z. Qin et al., "Semantic Communications: Principles and Challenges," arXiv:2212.00032, 2021.

[2] S. Sukhbaatar, A. Szlam, and J. Weston, "Learning Multiagent Communication with Backpropagation," arXiv:1605.07736, 2016.

[3] E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 3, pp. 567–579, 2019.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. CVPR, 2016, pp. 770–778.
