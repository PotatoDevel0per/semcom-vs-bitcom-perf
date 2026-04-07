# Semantic Communication vs. Raw-Bit Transmission

**14897 Topics in Mobile Computing — Mini-Project**
Spring 2026 | 이지혁 (20266243)

## 개요

무선 채널 제약 환경에서 **Semantic 통신**(의미적 표현 전송)과 **Raw-Bit 전송**(JPEG 압축 포함)의 성능을 정량적으로 비교한다. CIFAR-10 이미지 분류 태스크를 기준으로 다양한 전송 비트 예산 및 채널 조건에서 정확도, 효율성(Accuracy per bit), 강건성을 평가한다.

## 연구 문제

- 전송 자원이 제한된 환경에서 semantic 표현이 JPEG 대비 Accuracy per bit에서 우수한가?
- AWGN, Rayleigh 페이딩, 패킷 손실 조건에서 semantic 표현의 강건성은 어떻게 변하는가?

## 시스템 구성

```
[이미지] → Semantic Encoder → Quantizer → Channel → Classifier Head → [예측]
[이미지] → JPEG 압축        →            → Channel → ResNet-18      → [예측]  (baseline)
```

| 구성요소 | 내용 |
|----------|------|
| 데이터셋 | CIFAR-10 |
| Semantic Encoder | ResNet-18 기반, latent dim 64/128/256 |
| Quantizer | 8-bit scalar 또는 K-means vector quantization |
| 채널 모델 | AWGN / Rayleigh 페이딩 / 블록 단위 패킷 손실 |
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

# 2. Semantic Encoder 학습
python scripts/train_semantic.py --config configs/train_semantic.yaml

# 3. 평가 (Semantic)
python scripts/eval_semantic.py --config configs/default.yaml

# 4. 평가 (Baseline: Raw / JPEG)
python scripts/eval_baseline.py --config configs/default.yaml
```

## 평가 조건

| 항목 | 값 |
|------|----|
| 전송 비트 예산 | 256B, 512B, 1KB, 5KB, 50KB |
| SNR | 0, 5, 10, 20 dB |
| 패킷 손실률 | 1%, 5%, 10% |
| 반복 횟수 | 5회 (평균 ± 표준편차 보고) |

## 프로젝트 일정

| 항목 | 마감일 |
|------|--------|
| Project Proposal | 2026-03-18 ✓ |
| Progress Report | 2026-04-08 |
| Presentation | 2026-05-27 |
| Final Report | 2026-06-03 |

## 참고문헌

- Sukhbaatar et al. (2016). "Learning Multiagent Communication with Backpropagation." arXiv:1605.07736.
- Qin et al. (2021). "Semantic communications: Principles and challenges." arXiv:2212.00032.
