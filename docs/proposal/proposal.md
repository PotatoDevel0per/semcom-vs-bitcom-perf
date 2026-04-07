# Mini-Project Proposal
**과목:** Topics in Mobile Computing (14897)
**학번 / 이름:** 20266243 이지혁
**학과:** 컴퓨터공학과 (석사 1학기)
**제출일:** 2026-03-18

---

## 제목

**한국어:** Semantic 통신과 비트(원시) 통신의 성능 비교
**영문:** Performance Comparison of Semantic Communication vs. Raw-Bit Transmission

---

## Abstract (요약)

태스크 지향적 관점에서 송수신 간 전송 데이터의 표현 방식(원시 이미지 전송 vs. 의미적(semantic) 표현 전송)이 통신 자원 제약 하에서의 태스크 성능에 미치는 영향을 비교·분석한다. 특히 대역폭, 전송 지연 및 채널 손상(노이즈·페이딩·패킷 손실) 조건에서의 성능(정확도)·효율성(Accuracy per bit)·강건성(robustness)을 정량적으로 평가하는 것을 목표로 한다.

---

## 1. 연구동기 및 배경

6G 시대에서는 데이터 전송의 목적이 단순한 비트 전달을 넘어 "태스크 성능"을 직접 최적화하는 방향으로 변하고 있다. 이에 따라 송신측에서 태스크 관련 의미 정보를 추출하여 전송하는 semantic 통신의 가능성이 주목받고 있다. 본 연구는 semantic 통신이 실제 무선 채널 환경에서 원시 데이터 전송 대비 어떠한 이점을 제공하는지 실증적으로 규명하고자 한다.

---

## 2. 연구문제 (Research Questions)

- 전송 자원이 제한된 무선 환경에서 semantic 표현 전송이 원시 이미지 전송 대비 태스크 성능 대비 통신 효율(Accuracy per bit)에서 우수한가?
- 채널 잡음(SNR 변화), 페이딩, 패킷 손실 등 현실적 채널 제약 조건에서 semantic 표현의 강건성은 어떻게 변하는가?

---

## 3. 연구목표

- 다양한 전송비트 예산(256B, 512B, 1KB, 5KB, 50KB)에서 semantic 통신과 JPEG 기반 원시 전송의 분류 정확도 비교
- 채널 조건별(SNR: 0 / 5 / 10 / 20 dB, Rayleigh 페이딩, 패킷 손실률: 1% / 5% / 10%) 성능 평가
- Accuracy per bit, 전송 지연 추정, 통신비용 대비 성능 그래프 제시

---

## 4. 연구방법

### 4.1 데이터셋 및 태스크
- 기본 데이터셋: CIFAR-10 (이미지 분류)
- 확장 옵션: Tiny-ImageNet (시간 여유가 있을 경우)

### 4.2 모델 및 시스템 구성

| 구성요소 | 설명 |
|----------|------|
| 분류기 | ResNet-18 (기본 분류기 cell로 사용) |
| Semantic Encoder | ResNet 기반 변형 또는 소형 오토인코더를 활용하여 latent vector (64/128/256차원) 추출 |
| 압축/양자화 | 8-bit 양자화 및 간단한 벡터 양자화(K-means) 적용 |

### 4.3 채널 및 통신 모델
- 채널 모델: AWGN, Rayleigh 페이딩, 블록 단위 패킷 손실 시뮬레이션
- 전송비트 예산 설정에 따른 전송 메시지 길이 제한 적용

### 4.4 학습 및 평가 프로토콜
- Baseline: 원시 이미지 전송(무손실 이상적 가정) 및 JPEG 압축 전송 비교
- Semantic 방식: encoder(송신) + (선택적) decoder/수신 classifier 로 구성된 end-to-end 또는 encoder-only 학습
- 각 실험 설정별로 5회 이상 반복 수행하여 평균 및 표준편차 보고

---

## 5. 평가 지표

| 구분 | 세부 지표 |
|------|-----------|
| 주지표 | 분류 정확도 (Top-1 Accuracy) |
| 보조지표 | Accuracy per bit, 전송 바이트 수, 추정 전송 지연 (가정된 전송률 기반) |
| 강건성 지표 | SNR / 패킷 손실 변화에 따른 성능 저하율 |

---

## 6. 산출물

- **코드:** 실험 재현이 가능한 Git 저장소 (encoder, quantize, channel simulator, train/eval 스크립트)
- **실험 결과:** 그래프 (비트-성능 곡선, SNR 곡선), 통계표
- **보고서:** 중간보고서 (Progress report, 2026-04-08), 최종보고서 (IEEE conference format, 2026-06-03)
- **발표자료** (Presentation slides), 간단한 데모

---

## 7. 연구일정 (Gantt)

| 기간 | 주요 활동 | 산출물 / 마일스톤 |
|------|-----------|-------------------|
| 2026-03-13 ~ 03-17 | 환경 구축 및 baseline 학습 | - |
| **2026-03-18** | **Proposal 제출** | **제안서** |
| 2026-03-19 ~ 04-07 | Semantic encoder 개발 및 초기 실험 | - |
| **2026-04-08** | **Progress Report 제출** | **중간보고서** |
| 2026-04-09 ~ 05-20 | 확장 실험 및 통계 반복 | - |
| **2026-05-27** | **발표** | **발표자료** |
| **2026-06-03** | **최종보고서 제출** | **IEEE 포맷** |

---

## 8. 참고문헌

- Sukhbaatar, S., Szlam, A., & Weston, J. (2016). "Learning Multiagent Communication with Backpropagation" (CommNet). arXiv:1605.07736.
- Qin, Z., et al. (2021). Semantic communications: Principles and challenges. arXiv:2212.00032.
- 관련 recent works on semantic communications (arXiv / IEEE Transactions on Communications, IEEE JSAC).

---

## 9. 요구 자원 및 기타 사항

- **개발 환경:** Python, PyTorch
- **계산 자원:** GPU (로컬 또는 클라우드)
- **기타:** 데이터셋 및 중간 결과 공유를 위한 GitHub 레포지토리 사용
