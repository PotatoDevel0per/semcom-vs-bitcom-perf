# Semantic Communication vs. Raw-Bit Transmission

**14897 Topics in Mobile Computing — Mini-Project**
Spring 2026 | 이지혁 (20266243), 컴퓨터공학과 석사 1학기

---

## 목차

1. [개요](#1-개요)
2. [동기 및 배경](#2-동기-및-배경)
3. [연구 문제](#3-연구-문제)
4. [시스템 구조](#4-시스템-구조)
5. [방법론](#5-방법론)
6. [실험 설정](#6-실험-설정)
7. [실험 결과](#7-실험-결과)
8. [핵심 발견](#8-핵심-발견)
9. [한계 및 향후 연구](#9-한계-및-향후-연구)
10. [프로젝트 일정](#10-프로젝트-일정)
11. [실험 재현 방법](#11-실험-재현-방법)
12. [참고문헌](#12-참고문헌)

---

## 1. 개요

본 프로젝트는 무선 채널 제약 환경에서 **Semantic 통신**(태스크 관련 의미 표현을 전송하는 방식)과 **Raw-Bit 전송**(원시 픽셀 / JPEG 압축 전송)의 성능을 정량적으로 비교한다. CIFAR-10 이미지 분류를 하위 태스크로 두고, 다음 세 축으로 평가한다:

- **정확도** — Top-1 분류 정확도
- **통신 효율** — Accuracy per bit, 전송 바이트 수, 추정 전송 지연 (LoRa / LTE-M / 5G 가정)
- **강건성** — AWGN, Rayleigh 페이딩, 블록 단위 패킷 손실 조건에서의 정확도 유지율

추가로 **channel-naive vs channel-aware training** ablation을 통해 학습 시 채널 노출이 강건성에 미치는 영향을 분석했다.

## 2. 동기 및 배경

5G/6G 시대의 무선 통신은 단순 비트 전달을 넘어 **태스크 성능을 직접 최적화**하는 방향으로 진화하고 있으며, 이를 위한 패러다임으로 semantic communication이 주목받고 있다 [1, 7]. 송신측이 태스크 관련 의미 정보만을 추출·전송함으로써 페이로드를 크게 줄이면서도 수신측 태스크(분류, 인식, 의사결정 등) 성능을 유지하는 것이 목표다.

기존의 Shannon 기반 통신은 원시 비트의 무손실 복원을 목적으로 하므로, 하위 태스크와 무관하게 픽셀 수준 정보를 모두 전송한다. 제한된 대역폭·전력 환경(IoT, 모바일 edge inference)에서 이는 비효율적이며 [1, 7], JPEG·H.264 같은 source coding으로도 본질적 한계가 있다. DeepJSCC [2], DeepSC [8] 등은 송수신단의 deep encoder/decoder를 통해 source–channel coding을 종단 간(end-to-end) 학습함으로써 이 한계를 극복하려는 시도이다. 한편 task-oriented communication 관점에서 information bottleneck 기반 latent 압축 [4] 과 vector quantization 기반 codebook 전송 [5, 6] 이 활발히 연구되고 있다. 본 연구는 이 흐름의 image classification에 특화된 비교 분석에 해당한다.

## 3. 연구 문제

1. **통신 효율 비교** — 전송 자원이 제한된 무선 환경에서 semantic 표현이 JPEG 대비 Accuracy per bit에서 우수한가?
2. **채널 강건성** — AWGN, Rayleigh 페이딩, 패킷 손실 조건에서 semantic 표현의 강건성은 어떻게 변하는가?
3. **양자화 trade-off** — Scalar 양자화와 K-means 기반 vector 양자화는 어떤 비트-정확도 trade-off를 갖는가?
4. **학습 시 채널 노출의 영향** — Channel-aware training이 낮은 SNR에서의 성능에 어떤 영향을 주는가?

## 4. 시스템 구조

```
[Image] ─ Semantic Encoder ─ Quantizer ─┐
                                        ├─ Channel ─ Classifier Head ─ [Prediction]
[Image] ─ JPEG / Raw bit stream ────────┘                ResNet-18
```

| 구성요소 | 설명 | 참조 |
|----------|------|:---:|
| 데이터셋 | CIFAR-10 (50,000 train / 10,000 test, 32×32 RGB) | [11] |
| 분류기 (수신측) | ResNet-18 (CIFAR-10용 32×32 입력 적응) | [3] |
| Semantic Encoder | ResNet-18 backbone + 선형 projection (latent dim 64 / 128 / 256) | [3] |
| Quantizer | 8-bit Scalar uniform 또는 K-means Vector quantization (K ∈ {16, 64, 256, 1024}) | [5, 6] |
| 채널 모델 | AWGN / Rayleigh 페이딩 (Perfect CSI 등화) / 블록 단위 패킷 손실 | [2] |
| Baseline | Raw lossless 전송, JPEG 압축 (quality ∈ {1, 3, 5, 10, 20, 40, 60, 80, 95}) | — |

## 5. 방법론

### 5.1 분류기 baseline (ResNet-18)

CIFAR-10용으로 표준 ResNet-18 [3]의 첫 conv를 3×3 stride 1로 변경하고 첫 max-pool을 제거했다. SGD (lr=0.1, weight decay=5×10⁻⁴, momentum=0.9) + cosine annealing, 30 epochs 학습. 결과: **테스트 정확도 93.36%** (무손실 입력 상한).

### 5.2 Semantic Encoder + Classifier Head

`SemanticEncoder`는 분류기와 동일한 ResNet-18 백본의 feature extractor 끝에 선형 projection head를 두어 latent vector $z \in \mathbb{R}^d$ ($d \in \{64, 128, 256\}$)를 출력한다 [8]. `SemanticClassifierHead`는 2층 MLP (latent_dim → 256 → 10) 로 latent에서 직접 분류 logit을 산출한다. Encoder–head는 cross-entropy로 종단 간 학습되며, 학습 자체에는 quantization을 적용하지 않는다(평가 시 양자화 + 채널 적용). 본 설계는 task-oriented edge inference [4] 의 IB(Information Bottleneck) 관점과 일관된다.

### 5.3 Quantization

**Scalar Quantization (8-bit uniform)** — latent vector의 min/max 범위를 [0, 255]로 매핑한 uniform quantization. 전송 크기 = $d$ bytes.

**Vector Quantization (K-means)** [5, 6] — 학습된 encoder로 train set의 latent 분포를 추출한 뒤 MiniBatchKMeans로 codebook(K개 centroid)을 fitting. 전송 시 latent 전체를 codebook 인덱스(⌈log₂K⌉ bits)로 매핑. K ∈ {16, 64, 256, 1024} sweep.

### 5.4 채널 모델

**AWGN** — 신호 전력 측정 후 SNR_dB에 따라 정해진 noise variance의 Gaussian noise를 가산.

**Rayleigh fading** — Rayleigh 계수 $h \sim \mathcal{CN}(0, 1)$를 곱한 후 AWGN을 가산하고, **perfect CSI 등화** ($y / (h + \epsilon)$)로 복원. 단일 tap fading 가정.

**Block packet loss** — 블록 단위 Bernoulli mask로 latent dimension의 일부를 0으로 설정 (loss_rate ∈ {1%, 5%, 10%}).

### 5.5 Channel-aware Training (ablation)

학계 표준 SNR-adaptive training [2, 6, 8]을 따라, channel-aware 모델은 매 batch마다 SNR을 [0, 20] dB 균등 분포에서 무작위 샘플링해 latent에 AWGN을 적용한 뒤 head로 forward 한다:

```python
z = encoder(x)
snr_train = torch.empty(1).uniform_(0, 20).item()
z = AWGNChannel(snr_db=snr_train)(z)
logits = head(z)
```

Channel-naive 모델은 학습 시 채널을 적용하지 않는 기준선이다 (학계 weak baseline). 두 모델은 동일한 hyperparameter / epochs / seed로 학습된다.

## 6. 실험 설정

### 6.1 데이터셋

CIFAR-10 [11]: 50,000 train / 10,000 test, 32×32 RGB, 10 classes. 정규화 mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616). Train augmentation은 RandomCrop(32, padding=4) + RandomHorizontalFlip.

### 6.2 Hyperparameter

| 항목 | Classifier | Semantic Encoder |
|---|---|---|
| Optimizer | SGD (momentum=0.9) | Adam |
| LR | 0.1 (cosine) | 1×10⁻³ (cosine) |
| Weight decay | 5×10⁻⁴ | 1×10⁻⁴ |
| Epochs | 30 | 30 |
| Batch size | 128 | 128 |
| Seed | 42 | 42 |
| 학습 환경 | NVIDIA RTX A5000 × 2, PyTorch 2.5.1+cu121 | 동일 |

### 6.3 평가 프로토콜

- **반복**: 모든 sweep은 채널 noise seed를 변경하며 5회 반복, mean ± std 보고
- **SNR sweep**: AWGN/Rayleigh 각 0, 5, 10, 20 dB
- **Packet loss sweep**: 1%, 5%, 10%
- **VQ sweep**: K ∈ {16, 64, 256, 1024} (noiseless 채널, K-means codebook은 train latent로 fit)
- **Latency**: 전송 바이트 × 8 / 가정 link 전송률 (LoRa 1 kbps, LTE-M 1 Mbps, 5G 100 Mbps)

## 7. 실험 결과

### 7.1 학습된 모델 정확도

| 모델 | 전송 크기 | Channel-naive | Channel-aware (SNR sampling) |
|:-----|:---------:|:-------------:|:----------------------------:|
| ResNet-18 (분류기 baseline) | 3,072 B | 0.9336 | — |
| Semantic Encoder dim=64 | 64 B | 0.9224 | 0.9269 |
| Semantic Encoder dim=128 | 128 B | 0.9255 | 0.9233 |
| Semantic Encoder dim=256 | 256 B | 0.9213 | 0.9215 |

### 7.2 JPEG Baseline (분류기는 무손실 학습된 ResNet-18 재사용)

| Quality | 평균 바이트 | Top-1 Acc | Acc/Bit (×10⁻⁴) |
|:-------:|:-----------:|:---------:|:----------------:|
| q=1   | 654.6 B   | 0.2013 | 0.38 |
| q=10  | 699.5 B   | 0.4354 | 0.78 |
| q=20  | 741.5 B   | 0.5997 | 1.01 |
| q=40  | 801.3 B   | 0.7301 | 1.14 |
| q=60  | 853.8 B   | 0.7949 | **1.16** |
| q=80  | 960.6 B   | 0.8556 | 1.11 |
| q=95  | 1,300.8 B | 0.9142 | 0.88 |

→ JPEG는 q=60에서 Acc/bit 최고, 91% 정확도를 위해서는 1,300 B 필요.

### 7.3 AWGN 채널 — SNR sweep (Scalar 8-bit, 5회 반복)

| dim | 0 dB | 5 dB | 10 dB | 20 dB |
|:---:|:----:|:----:|:-----:|:-----:|
| 64  | 0.8866 ± 0.0009 | 0.9112 ± 0.0007 | 0.9184 ± 0.0010 | 0.9214 ± 0.0003 |
| 128 | 0.9075 ± 0.0013 | 0.9203 ± 0.0011 | 0.9233 ± 0.0008 | **0.9255 ± 0.0002** |
| 256 | 0.9132 ± 0.0006 | 0.9185 ± 0.0005 | 0.9202 ± 0.0005 | 0.9212 ± 0.0003 |

### 7.4 Rayleigh 페이딩 — SNR sweep (Perfect CSI, 5회 반복)

| dim | 0 dB | 5 dB | 10 dB | 20 dB |
|:---:|:----:|:----:|:-----:|:-----:|
| 64  | 0.8842 ± 0.0038 | 0.9112 ± 0.0020 | 0.9181 ± 0.0014 | 0.9216 ± 0.0007 |
| 128 | 0.9060 ± 0.0022 | 0.9182 ± 0.0024 | 0.9220 ± 0.0014 | 0.9252 ± 0.0008 |
| 256 | **0.9113 ± 0.0021** | 0.9177 ± 0.0017 | 0.9196 ± 0.0008 | 0.9210 ± 0.0006 |

AWGN 대비 정확도 손실은 0.3%p 미만이나, std는 약 2~5배 증가 (페이딩 계수 무작위성).

### 7.5 블록 단위 패킷 손실 (5회 반복)

| dim | 손실률 1% | 손실률 5% | 손실률 10% |
|:---:|:--------:|:--------:|:---------:|
| 64  | 0.9221 ± 0.0003 | 0.9216 ± 0.0003 | 0.9210 ± 0.0007 |
| 128 | 0.9251 ± 0.0004 | 0.9244 ± 0.0005 | 0.9237 ± 0.0007 |
| 256 | 0.9215 ± 0.0002 | 0.9212 ± 0.0006 | 0.9208 ± 0.0007 |

10% 손실 환경에서도 dim=128 모델은 92.37% 유지 (무손실 학습 baseline 대비 −0.99%p).

### 7.6 Vector Quantization (K-means, noiseless)

| dim | K=16 (0.5 B) | K=64 (0.75 B) | K=256 (1 B) | K=1024 (1.25 B) |
|:---:|:------------:|:-------------:|:-----------:|:---------------:|
| 64  | 0.9011 | 0.9187 | 0.9194 | 0.9196 |
| 128 | 0.9022 | **0.9220** | 0.9220 | 0.9193 |
| 256 | 0.9015 | 0.9193 | 0.9165 | 0.9173 |

→ **dim=128 + K=64 codebook → 단 0.75 B (= 6 bits) 로 92.20% 정확도**. JPEG q=95 (1,300 B, 91.42%) 와 동등 정확도에서 약 **1,700배 적은 대역폭**. 이는 Kutay & Yener [5] 및 VQ-DeepSC [6] 가 보고한 vector quantization의 효율성과 부합한다.

### 7.7 Channel-aware Training Ablation

**AWGN — Channel-aware 정확도 (Δ = aware − naive)**

| dim | 0 dB | 5 dB | 10 dB | 20 dB |
|:---:|:-----|:-----|:------|:------|
| 64  | 0.9149 ± 0.0013 (**+2.83**) | 0.9227 ± 0.0011 (+1.15) | 0.9250 ± 0.0007 (+0.66) | 0.9265 ± 0.0002 (+0.51) |
| 128 | 0.9148 ± 0.0005 (+0.73)     | 0.9202 ± 0.0009 (−0.01) | 0.9221 ± 0.0005 (−0.12) | 0.9230 ± 0.0004 (−0.25) |
| 256 | 0.9180 ± 0.0009 (+0.48)     | 0.9211 ± 0.0008 (+0.26) | 0.9220 ± 0.0005 (+0.18) | 0.9217 ± 0.0001 (+0.05) |

**Rayleigh — Channel-aware 정확도 (Δ = aware − naive)**

| dim | 0 dB | 5 dB | 10 dB | 20 dB |
|:---:|:-----|:-----|:------|:------|
| 64  | 0.9145 ± 0.0017 (**+3.03**) | 0.9230 ± 0.0009 (+1.18) | 0.9251 ± 0.0003 (+0.70) | 0.9263 ± 0.0003 (+0.47) |
| 128 | 0.9153 ± 0.0007 (+0.93)     | 0.9208 ± 0.0002 (+0.26) | 0.9228 ± 0.0005 (+0.08) | 0.9235 ± 0.0003 (−0.17) |
| 256 | 0.9178 ± 0.0010 (+0.65)     | 0.9211 ± 0.0004 (+0.34) | 0.9215 ± 0.0004 (+0.19) | 0.9218 ± 0.0004 (+0.08) |

→ **dim=64 + 저SNR (0 dB)** 조합에서 효과 최대화. AWGN 0 dB에서 **+2.83%p**, Rayleigh 0 dB에서 **+3.03%p**.
→ 고 SNR (20 dB)에서는 dim=128에 한해 −0.25%p 미미한 trade-off — 학습 분포 변화의 자연스러운 비용.
→ **Channel-aware dim=64 (64 B) > Channel-naive dim=128 (128 B)** — AWGN/Rayleigh 모든 SNR에서. 작은 latent도 다양한 채널 노출만 있으면 더 강건. 이는 DeepJSCC [2] 와 DeepSC [8] 가 보고한 channel-aware 학습 효과와 일치한다.

### 7.8 추정 전송 지연 (가정된 link rate 기반)

| 방식 | 평균 바이트 | Top-1 | LoRa (1 kbps) | LTE-M (1 Mbps) | 5G (100 Mbps) |
|:-----|:-----------:|:-----:|:-------------:|:--------------:|:-------------:|
| Raw (lossless) | 3,072 B | 93.36% | 24.58 s | 24.58 ms | 246 µs |
| JPEG q=60 | 853.8 B | 79.49% | 6.83 s | 6.83 ms | 68.3 µs |
| JPEG q=95 | 1,300.8 B | 91.42% | 10.41 s | 10.41 ms | 104 µs |
| Semantic Scalar dim=64 (AWGN 10 dB) | 64 B | 91.84% | 512 ms | 512 µs | 5.1 µs |
| Semantic Scalar dim=128 (AWGN 10 dB) | 128 B | 92.33% | 1.02 s | 1.02 ms | 10.2 µs |
| **Semantic VQ dim=128, K=64** | **0.75 B** | **92.20%** | **6 ms** | **6 µs** | **0.06 µs** |

→ **LoRa 환경에서 Raw 24.58 s → Semantic VQ 6 ms** (약 4,100× 단축).
→ **5G 환경에서 Semantic VQ 60 ns** — 실시간 추론 파이프라인의 네트워크 병목 사실상 제거.

### 7.9 통신 효율 종합 (Accuracy per Bit)

| 방식 | 바이트 | Top-1 | Acc/Bit |
|:-----|:------:|:-----:|:-------:|
| JPEG q=60 (JPEG 중 최고 Acc/bit) | 853.8 B | 0.7949 | 1.16×10⁻⁴ |
| JPEG q=95 (JPEG 중 최고 정확도) | 1,300.8 B | 0.9142 | 0.88×10⁻⁴ |
| Semantic Scalar dim=64 (AWGN 10 dB) | 64 B | 0.9184 | 17.9×10⁻⁴ |
| Semantic Scalar dim=128 (AWGN 20 dB) | 128 B | 0.9255 | 9.04×10⁻⁴ |
| **Semantic Vector dim=128, K=64** | **0.75 B** | **0.9220** | **0.1537** |

→ Vector quantization은 JPEG q=60 대비 Acc/Bit 기준 약 **1,325× 우수**. 이는 image-classification 태스크에 한정된 결과지만, task-oriented edge inference [4] 관점에서는 IB(Information Bottleneck) 이론과 일관된 경향이다.

## 8. 핵심 발견

1. **Vector Quantization은 압도적 효율** — dim=128 + K=64 codebook의 6-bit 전송에서 92.20% 정확도. JPEG가 동등 정확도를 내려면 약 1,700× 큰 페이로드 필요.
2. **저대역폭 IoT에서 지연 격차 극대화** — LoRa 가정 시 Raw 24.58 s vs Semantic VQ 6 ms (4,100×). 5G에서도 VQ는 60 ns 수준으로 네트워크 병목 사실상 제거.
3. **Channel-aware training은 저SNR에서 결정적** — dim=64 모델은 AWGN 0 dB에서 +2.83%p, Rayleigh 0 dB에서 +3.03%p 향상. Channel-aware dim=64가 channel-naive dim=128보다 모든 SNR에서 우수 — 작은 latent도 다양한 채널 노출로 강건성 확보 가능.
4. **Rayleigh 페이딩도 강건** — Perfect CSI 등화 가정 하에 AWGN 대비 평균 정확도 손실 0.3%p 미만 (분산은 2~5× 증가).
5. **dim별 trade-off** — 큰 latent (256)는 낮은 SNR에 더 강건, 작은 latent (64)는 acc/bit 효율이 우수. Channel-aware training 도입 시 dim=64로도 충분한 강건성 확보.
6. **결과 재현성 우수** — 5회 반복 std 대부분 < 0.002.

## 9. 한계 및 향후 연구

**한계:**
- 단일 모달리티(이미지) + 단일 태스크(분류). 의미 표현의 일반화는 입증되지 않음.
- Rayleigh 평가에서 **perfect CSI 등화 가정**. 실제 통신 시스템의 imperfect CSI / pilot overhead는 미반영.
- Vector quantization 평가는 **noiseless 채널 가정**. 양자화 인덱스 자체의 채널 강건성은 별도 분석 필요.
- 비교 baseline은 JPEG 한정. JPEG2000 / WebP / learned image codec 비교는 미수행.

**향후 연구:**
- **Multi-task semantic communication** — 단일 latent로 분류 + 분할 + 검색 등 복수 태스크 동시 지원 [9]
- **Channel-aware VQ** — codebook 자체를 채널 통계에 적응시키는 joint design [10]
- **Modality 확장** — 텍스트 / 음성 / 멀티모달로의 확장 (DeepSC [8] 의 확장 방향)
- **Imperfect CSI** — Pilot 기반 채널 추정 오차를 포함한 평가
- **데이터셋 확장** — Tiny-ImageNet, ImageNet으로 일반화 입증

## 10. 프로젝트 일정

| 항목 | 마감일 | 상태 |
|------|--------|:----:|
| Project Proposal | 2026-03-18 | ✓ |
| Progress Report | 2026-04-08 | ✓ |
| Presentation | 2026-05-27 | 예정 |
| Final Report | 2026-06-03 | 예정 |

---

## 11. 실험 재현 방법

### 11.1 환경 설정

```bash
# 가상환경 생성 및 활성화 (Python 3.10 이상)
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# CUDA 12.x 드라이버 환경에서는 PyTorch cu121 빌드로 재설치 권장
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

`configs/default.yaml` 의 `device` 필드는 `"cuda"` (NVIDIA GPU) / `"cpu"` / `"mps"` (Apple Silicon) 중 환경에 맞게 설정.

### 11.2 실행 순서

```bash
# 1. ResNet-18 분류기 사전학습
python scripts/train_classifier.py --config configs/train_classifier.yaml

# 2a. Semantic Encoder 학습 (channel-naive)
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 64
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 128
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 256

# 2b. Semantic Encoder 학습 (channel-aware, random SNR ∈ [0, 20] dB sampling)
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 64  --channel_aware
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 128 --channel_aware
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 256 --channel_aware

# 3. 평가 (semantic naive+aware, baseline)
python scripts/eval_semantic.py --config configs/default.yaml
python scripts/eval_baseline.py --config configs/default.yaml

# 4. 그래프 생성
python scripts/plot_results.py

# 5. 추정 전송 지연 계산 (LoRa / LTE-M / 5G 가정)
python scripts/compute_latency.py
```

### 11.3 학습 시간 (참고)

NVIDIA RTX A5000 1장 기준 — Classifier 30 epochs ≈ 7분, Semantic Encoder 30 epochs ≈ 6분/dim. 평가 (5회 반복 × naive + aware × 3 dim × 모든 sweep) ≈ 6분.

### 11.4 저장소 구조

```
.
├── configs/                 # YAML 실험 설정
├── docs/                    # proposal, progress report (LaTeX 포함)
├── experiments/
│   ├── checkpoints/         # *.pth (gitignored)
│   ├── results/             # *.json (학습/평가 결과)
│   └── figures/             # *.png (gitignored)
├── scripts/                 # train / eval / plot / latency 스크립트
└── src/
    ├── data/                # CIFAR-10 loader + transforms
    ├── models/              # ResNet-18 classifier, Semantic encoder/head, Quantizer
    ├── channel/             # AWGN, Rayleigh, PacketLoss
    ├── baseline/            # Raw, JPEG
    ├── metrics/             # Top-1, Accuracy per bit
    └── utils/               # seed, logger, checkpoint
```

---

## 12. 참고문헌

[1] Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, "Semantic Communications: Principles and Challenges," arXiv:2201.01389, 2021.

[2] E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE Transactions on Cognitive Communications and Networking*, vol. 5, no. 3, pp. 567–579, 2019.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[4] J. Shao, Y. Mao, and J. Zhang, "Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach," *IEEE Journal on Selected Areas in Communications*, vol. 40, no. 1, pp. 197–211, 2022. *(arXiv:2102.04170, 2021)*

[5] M. Kutay and A. Yener, "Vector Quantized Semantic Communication for Image Classification," 2024.

[6] Q. Hu, G. Zhang, Z. Bao, Z. Lin, and F.-Y. Wang, "Robust Semantic Communications with Masked VQ-VAE Enabled Codebook," arXiv:2206.04011, 2022. *(VQ-DeepSC)*

[7] E. C. Strinati and S. Barbarossa, "6G Networks: Beyond Shannon Towards Semantic and Goal-Oriented Communications," *Computer Networks*, vol. 190, p. 107930, 2021.

[8] H. Xie, Z. Qin, G. Y. Li, and B. H. Juang, "Deep Learning Enabled Semantic Communication Systems," *IEEE Transactions on Signal Processing*, vol. 69, pp. 2663–2675, 2021. *(DeepSC)*

[9] X. Lyu et al., "Multi-Task Semantic Communications," 2023.

[10] X. Meng et al., "Channel-Aware Vector Quantization for Semantic Communications," 2025.

[11] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, University of Toronto, 2009. *(CIFAR-10 dataset)*

> 본문의 [n] 표기는 위 번호와 일치합니다. 본문에서 인용된 위치 요약:
> - **§2 동기/배경**: [1, 7] 6G 패러다임·motivation / [2, 8] DeepJSCC·DeepSC 흐름 / [4] task-oriented IB / [5, 6] VQ
> - **§4 시스템 구조**: [3] ResNet, [5, 6] VQ, [2] 채널 모델, [11] CIFAR-10
> - **§5.1 분류기**: [3]
> - **§5.2 Encoder**: [4, 8]
> - **§5.3 VQ**: [5, 6]
> - **§5.5 Channel-aware**: [2, 6, 8]
> - **§6.1 데이터셋**: [11]
> - **§7.6 VQ 결과**: [5, 6]
> - **§7.7 Channel-aware 결과**: [2, 8]
> - **§7.9 효율 비교**: [4]
> - **§9 향후 연구**: [9] multi-task, [10] channel-aware VQ, [8] modality 확장
