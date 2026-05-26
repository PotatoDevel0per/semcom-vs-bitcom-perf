# Semantic Communication vs. Raw-Bit Transmission

**14897 Topics in Mobile Computing — Mini-Project**
Spring 2026 | 이지혁 (20266243)

## 개요

무선 채널 제약 환경에서 **Semantic 통신**(의미적 표현 전송)과 **Raw-Bit 전송**(JPEG 압축 포함)의 성능을 정량적으로 비교한다. CIFAR-10 이미지 분류 태스크를 기준으로 다양한 전송 비트 예산 및 채널 조건에서 정확도, 효율성(Accuracy per bit), 강건성을 평가한다.

## 연구 문제

- 전송 자원이 제한된 환경에서 semantic 표현이 JPEG 대비 Accuracy per bit에서 우수한가?
- AWGN, Rayleigh 페이딩, 패킷 손실 조건에서 semantic 표현의 강건성은 어떻게 변하는가?
- Scalar 양자화와 Vector 양자화(K-means)는 어떤 trade-off를 갖는가?

## 시스템 구성

```
[이미지] → Semantic Encoder → Quantizer → Channel → Classifier Head → [예측]
[이미지] → JPEG 압축        →            → Channel → ResNet-18      → [예측]  (baseline)
```

| 구성요소 | 내용 |
|----------|------|
| 데이터셋 | CIFAR-10 |
| Semantic Encoder | ResNet-18 기반, latent dim 64/128/256 |
| Quantizer | 8-bit scalar quantization 또는 K-means vector quantization (K ∈ {16, 64, 256, 1024}) |
| 채널 모델 | AWGN / Rayleigh 페이딩 (Perfect CSI) / 블록 단위 패킷 손실 |
| Baseline | 원시 전송(무손실), JPEG 압축 전송 |

## 환경 설정

```bash
# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치 (최초 1회)
pip install -r requirements.txt
```

## 실행 방법

```bash
# 1. 분류기 사전학습
python scripts/train_classifier.py --config configs/train_classifier.yaml

# 2a. Semantic Encoder 학습 (channel-naive)
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 64
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 128
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 256

# 2b. Channel-aware Semantic Encoder 학습 (random SNR ∈ [0, 20] dB sampling)
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 64  --channel_aware
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 128 --channel_aware
python scripts/train_semantic.py --config configs/train_semantic.yaml --latent_dim 256 --channel_aware

# 3. 평가 (Semantic + Baseline)
python scripts/eval_semantic.py --config configs/default.yaml
python scripts/eval_baseline.py --config configs/default.yaml

# 4. 그래프 생성
python scripts/plot_results.py

# 5. 추정 전송 지연 계산 (LoRa / LTE-M / 5G 가정)
python scripts/compute_latency.py
```

## 평가 조건

| 항목 | 값 |
|------|----|
| 전송 비트 예산 | 256 B, 512 B, 1 KB, 5 KB, 50 KB |
| SNR (AWGN/Rayleigh) | 0, 5, 10, 20 dB |
| 패킷 손실률 | 1%, 5%, 10% |
| 반복 횟수 | 5회 (평균 ± 표준편차 보고) |

---

## 실험 결과

> 학습 환경: NVIDIA RTX A5000 × 2, PyTorch 2.5.1+cu121, Python 3.11
> 모든 모델은 동일한 hyperparameter로 30 epochs 학습. 평가는 채널 noise seed를 변경해 5회 반복.

### 1. 학습된 모델 정확도

두 학습 방식을 비교: **channel-naive** (학습 시 채널 미적용) 와 **channel-aware** (매 batch마다 AWGN SNR ∈ [0, 20] dB 무작위 샘플링).

| 모델 | 전송 크기 | Channel-naive | Channel-aware (SNR sampling) |
|:-----|:---------:|:-------------:|:----------------------------:|
| ResNet-18 (분류기 baseline) | 3,072 B | 0.9336 | — |
| Semantic Encoder dim=64 | 64 B | 0.9224 | 0.9269 |
| Semantic Encoder dim=128 | 128 B | 0.9255 | 0.9233 |
| Semantic Encoder dim=256 | 256 B | 0.9213 | 0.9215 |

→ Noise-free 평가에서는 거의 동일하나, **낮은 SNR 환경에서 channel-aware가 큰 폭으로 우세** (§5.5 참조).

### 2. JPEG Baseline (분류기는 무손실 학습된 ResNet-18 재사용)

| Quality | 평균 바이트 | Top-1 Acc | Acc/Bit (×10⁻⁴) |
|:-------:|:-----------:|:---------:|:----------------:|
| q=1   | 654.6 B   | 0.2013 | 0.38 |
| q=10  | 699.5 B   | 0.4354 | 0.78 |
| q=20  | 741.5 B   | 0.5997 | 1.01 |
| q=40  | 801.3 B   | 0.7301 | 1.14 |
| q=60  | 853.8 B   | 0.7949 | 1.16 |
| q=80  | 960.6 B   | 0.8556 | 1.11 |
| q=95  | 1,300.8 B | 0.9142 | 0.88 |

→ JPEG는 q=60에서 Acc/bit 최고(1.16×10⁻⁴), 91% 정확도 도달에 1,300 B 필요.

### 3. AWGN 채널 — SNR sweep (Scalar 8-bit, 5회 반복 mean ± std)

| latent dim | 0 dB | 5 dB | 10 dB | 20 dB |
|:----------:|:----:|:----:|:-----:|:-----:|
| 64  | 0.8866 ± 0.0009 | 0.9112 ± 0.0007 | 0.9184 ± 0.0010 | 0.9214 ± 0.0003 |
| 128 | 0.9075 ± 0.0013 | 0.9203 ± 0.0011 | 0.9233 ± 0.0008 | **0.9255 ± 0.0002** |
| 256 | 0.9132 ± 0.0006 | 0.9185 ± 0.0005 | 0.9202 ± 0.0005 | 0.9212 ± 0.0003 |

### 4. Rayleigh 페이딩 — SNR sweep (Perfect CSI, 5회 반복)

| latent dim | 0 dB | 5 dB | 10 dB | 20 dB |
|:----------:|:----:|:----:|:-----:|:-----:|
| 64  | 0.8842 ± 0.0038 | 0.9112 ± 0.0020 | 0.9181 ± 0.0014 | 0.9216 ± 0.0007 |
| 128 | 0.9060 ± 0.0022 | 0.9182 ± 0.0024 | 0.9220 ± 0.0014 | 0.9252 ± 0.0008 |
| 256 | **0.9113 ± 0.0021** | 0.9177 ± 0.0017 | 0.9196 ± 0.0008 | 0.9210 ± 0.0006 |

→ 동일 SNR에서 AWGN 대비 정확도 손실 < 0.3%p. 분산은 AWGN보다 약 2~5배 큼 (페이딩 계수 무작위성 반영).

### 5. 블록 단위 패킷 손실 (5회 반복)

| latent dim | 손실률 1% | 손실률 5% | 손실률 10% |
|:----------:|:--------:|:--------:|:---------:|
| 64  | 0.9221 ± 0.0003 | 0.9216 ± 0.0003 | 0.9210 ± 0.0007 |
| 128 | 0.9251 ± 0.0004 | 0.9244 ± 0.0005 | 0.9237 ± 0.0007 |
| 256 | 0.9215 ± 0.0002 | 0.9212 ± 0.0006 | 0.9208 ± 0.0007 |

→ 10% 손실 환경에서도 dim=128 모델은 정확도 92.37% 유지 (무손실 학습 baseline 대비 0.99%p 손실).

### 6. Vector Quantization (K-means, noiseless 채널)

K-means codebook을 train set의 latent 분포에 fit한 후, test 시 인덱스만 전송. 전송 크기 = ⌈log₂K⌉ / 8 bytes (latent 전체를 하나의 인덱스로 매핑).

| latent dim | K=16 (0.5 B) | K=64 (0.75 B) | K=256 (1 B) | K=1024 (1.25 B) |
|:----------:|:------------:|:-------------:|:-----------:|:---------------:|
| 64  | 0.9011 | 0.9187 | 0.9194 | 0.9196 |
| 128 | 0.9022 | **0.9220** | 0.9220 | 0.9193 |
| 256 | 0.9015 | 0.9193 | 0.9165 | 0.9173 |

→ **dim=128 + K=64 codebook → 단 0.75 B (= 6 bits)로 92.20% 정확도.**
→ JPEG q=95 (1,300 B, 91.42%)와 정확도 동등하면서 약 **1,700배 적은 대역폭**.

### 6.5. Ablation: Channel-naive vs Channel-aware Training

학계 표준의 **SNR-adaptive training**(매 batch마다 AWGN SNR ∈ [0, 20] dB 무작위 샘플링)을 도입한 후 동일 sweep에서 비교. 모든 평가는 5회 반복 mean ± std.

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

→ **dim=64 + 저SNR (0 dB)** 조합에서 효과가 가장 크며, AWGN 0 dB에서 **+2.83%p**, Rayleigh 0 dB에서 **+3.03%p** 향상.
→ 고 SNR (20 dB)에서는 dim=128에 한해 −0.25%p 미미한 trade-off — 학습 분포의 변화에 따른 자연스러운 비용. 다른 dim/SNR 조합에서는 모두 향상 또는 보존.
→ Channel-aware dim=64는 AWGN/Rayleigh 모든 SNR에서 channel-naive dim=128보다 우수 — **작은 latent도 다양한 채널 노출로 더 강건한 표현을 학습**함을 시사.

### 7. 추정 전송 지연 (Estimated Transmission Latency)

각 방식의 평균 페이로드 크기에 가정된 전송률을 적용해 단일 이미지 전송 지연을 계산. Proposal §5 보조지표.

| 방식 | 평균 바이트 | Top-1 Acc | LoRa (1 kbps) | LTE-M (1 Mbps) | 5G (100 Mbps) |
|:-----|:-----------:|:---------:|:-------------:|:--------------:|:-------------:|
| Raw (lossless) | 3,072 B | 93.36% | 24.58 s | 24.58 ms | 246 µs |
| JPEG q=60 | 853.8 B | 79.49% | 6.83 s | 6.83 ms | 68.3 µs |
| JPEG q=95 | 1,300.8 B | 91.42% | 10.41 s | 10.41 ms | 104 µs |
| Semantic Scalar dim=64 (AWGN 10dB) | 64 B | 91.84% | 512 ms | 512 µs | 5.1 µs |
| Semantic Scalar dim=128 (AWGN 10dB) | 128 B | 92.33% | 1.02 s | 1.02 ms | 10.2 µs |
| **Semantic VQ dim=128, K=64** | **0.75 B** | **92.20%** | **6 ms** | **6 µs** | **0.06 µs** |

→ **LoRa 환경에서 Raw 24.58 s → Semantic VQ 6 ms** (약 **4,100배 단축**).
→ **5G 환경에서 Semantic VQ는 60 ns** 수준으로, 실시간 추론 파이프라인의 네트워크 병목을 거의 제거.

### 8. 통신 효율 비교 (Accuracy per Bit)

| 방식 | 바이트 | Top-1 Acc | Acc/Bit |
|:-----|:------:|:---------:|:-------:|
| JPEG q=60 (JPEG 중 최고 Acc/bit) | 853.8 B | 0.7949 | 1.16×10⁻⁴ |
| JPEG q=95 (JPEG 중 최고 acc) | 1,300.8 B | 0.9142 | 0.88×10⁻⁴ |
| Semantic Scalar dim=64 (AWGN 10dB) | 64 B | 0.9184 | 17.9×10⁻⁴ |
| Semantic Scalar dim=128 (AWGN 20dB) | 128 B | 0.9255 | 9.04×10⁻⁴ |
| **Semantic Vector dim=128, K=64** | **0.75 B** | **0.9220** | **0.1537** |

→ Vector quantization은 JPEG q=60 대비 Acc/Bit 기준 약 **1,325배 우수**.

---

## 핵심 발견

1. **Semantic + Vector quantization은 압도적 효율**: dim=128 + K=64 codebook으로 **6 bits 전송**에서 92.20% 정확도. JPEG가 동등 정확도를 내려면 약 1,700배 큰 페이로드가 필요하다.
2. **저대역폭 IoT에서 지연 시간 격차 극대화**: LoRa(1 kbps) 가정에서 Raw 24.58 s → Semantic VQ **6 ms** (약 4,100배 단축). 5G에서도 Semantic VQ는 60 ns 수준으로 네트워크 병목이 사실상 사라진다.
3. **Channel-aware training은 저SNR에서 결정적**: dim=64 모델은 AWGN 0 dB에서 +2.83%p, Rayleigh 0 dB에서 +3.03%p 향상. **Channel-aware dim=64가 channel-naive dim=128보다 모든 SNR에서 우수** — 작은 latent도 다양한 채널 노출로 강건성 확보 가능.
4. **Rayleigh 페이딩도 강건**: AWGN 대비 분산은 커지지만(2~5배), 평균 정확도 손실은 0.3%p 미만 (Perfect CSI 등화 가정).
5. **dim별 trade-off**: 큰 latent (256)는 낮은 SNR에 더 강건, 작은 latent (64)는 acc/bit 효율이 우수. Channel-aware 도입 시 dim=64로도 충분.
6. **5회 반복 std 매우 작음** (대부분 < 0.002): 채널 noise random성의 영향이 적고 결과 재현성이 매우 높다.

자세한 그래프는 [experiments/figures/](experiments/figures/) 디렉토리 참조.

---

## 프로젝트 일정

| 항목 | 마감일 | 상태 |
|------|--------|------|
| Project Proposal | 2026-03-18 | ✓ |
| Progress Report | 2026-04-08 | ✓ |
| Presentation | 2026-05-27 | 예정 |
| Final Report | 2026-06-03 | 예정 |

## 참고문헌

- Sukhbaatar et al. (2016). "Learning Multiagent Communication with Backpropagation." arXiv:1605.07736.
- Qin et al. (2021). "Semantic communications: Principles and challenges." arXiv:2212.00032.
- Bourtsoulatze et al. (2019). "Deep Joint Source-Channel Coding for Wireless Image Transmission." IEEE Trans. Cogn. Commun. Netw., 5(3), 567–579.
- He et al. (2016). "Deep Residual Learning for Image Recognition." Proc. CVPR, 770–778.
