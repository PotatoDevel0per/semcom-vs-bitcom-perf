# TMC Mini-Project — CLAUDE.md

## 프로젝트 개요

**과목:** 14897 Topics in Mobile Computing (Spring 2026)
**주제:** Semantic 통신과 비트(원시) 통신의 성능 비교
**학번/이름:** 20266243 이지혁
**유형:** 성능 비교형 (Option a)

---

## 마감 일정

| 항목 | 마감일 | 상태 |
|------|--------|------|
| Project Proposal | 2026-03-18 | 완료 |
| Progress Report | 2026-04-08 | 진행 중 |
| Presentation | 2026-05-27 | 예정 |
| Final Report | 2026-06-03 | 예정 |

---

## 환경 설정

```bash
# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치 (최초 1회)
pip install -r requirements.txt
```

**Python:** 3.x (`.venv/`)
**주요 패키지:** PyTorch, torchvision, scikit-learn, omegaconf, matplotlib, pandas

---

## 디렉토리 구조

```
TMC-MiniProject/
├── configs/            # YAML 실험 설정 (default, train_classifier, train_semantic)
├── docs/
│   ├── professor/      # 교수님 미니프로젝트 개요
│   └── proposal/       # 제출한 프로젝트 제안서
├── experiments/
│   ├── checkpoints/    # 학습된 모델 가중치 (.pth) — git 미포함
│   ├── results/        # 실험 결과 CSV — git 미포함
│   ├── figures/        # 그래프 이미지 — git 미포함
│   └── logs/           # 학습 로그 — git 미포함
├── notebooks/          # 탐색적 분석 및 최종 그래프 생성
├── scripts/            # 실행 진입점 (train/eval 스크립트)
├── src/
│   ├── data/           # CIFAR-10 DataLoader, transforms
│   ├── models/         # ResNet-18 분류기, SemanticEncoder/Decoder, Quantizer
│   ├── channel/        # AWGN, Rayleigh, PacketLoss 채널 시뮬레이터
│   ├── baseline/       # Raw 전송, JPEG 압축 baseline
│   ├── metrics/        # Top-1 Accuracy, Accuracy per bit 평가
│   └── utils/          # seed, logger, checkpoint 유틸
└── tests/              # pytest 단위 테스트
```

---

## 실험 실행 순서

```bash
# 1. ResNet-18 분류기 사전학습
python scripts/train_classifier.py --config configs/train_classifier.yaml

# 2. Semantic Encoder 학습 (latent_dim: 64 / 128 / 256)
python scripts/train_semantic.py --config configs/train_semantic.yaml

# 3. Semantic 방식 평가
python scripts/eval_semantic.py --config configs/default.yaml

# 4. Baseline 평가 (Raw / JPEG)
python scripts/eval_baseline.py --config configs/default.yaml
```

---

## 핵심 설계 원칙

- **재현성:** 모든 스크립트는 `set_seed(config.seed)`를 첫 줄에 호출한다. 기본 seed=42.
- **설정 관리:** `configs/default.yaml`이 단일 진실의 원천. 실험별 YAML이 이를 override.
- **결과 저장:** 수치 결과는 CSV, 그래프는 PNG/PDF로 `experiments/` 하위에 저장.
- **반복 실험:** 각 실험 조건은 5회 반복 후 평균/표준편차를 보고한다.

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Top-1 Accuracy | 분류 정확도 (주지표) |
| Accuracy per bit | 전송 비트당 정확도 (효율성) |
| SNR sweep | 0 / 5 / 10 / 20 dB 조건별 성능 |
| 패킷 손실 sweep | 1% / 5% / 10% 손실률별 성능 |
| 전송 비트 예산 | 256B / 512B / 1KB / 5KB / 50KB |

---

## 보고서 요건 (IEEE conference paper format)

- **Progress Report:** 현재까지 완료된 작업 + 업데이트된 Gantt chart
- **Final Report:** title / abstract / keywords / introduction / related works / methodology / performance evaluation / conclusions / references
- **참고문헌:** 10개 이상, 모두 본문에서 최소 1회 인용
