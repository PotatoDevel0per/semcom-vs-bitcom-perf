# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**과목:** 14897 Topics in Mobile Computing (Spring 2026)
**주제:** Semantic 통신 vs. Raw-Bit 전송 성능 비교 (CIFAR-10 분류 태스크 기준)
**유형:** 성능 비교형 (Option a)
**산출물:** Proposal (완료) → Progress Report → Presentation (2026-05-27) → Final Report (2026-06-03)

---

## 환경 / 실행

```bash
source .venv/bin/activate
pip install -r requirements.txt   # 최초 1회
```

스크립트는 **모듈이 아닌 파일 직접 실행** 방식이며, 각 파일 상단에서 `sys.path.insert`로 레포 루트를 path에 주입한다. 따라서 항상 레포 루트에서 다음과 같이 실행:

```bash
# 1. ResNet-18 분류기 사전학습
python scripts/train_classifier.py --config configs/train_classifier.yaml

# 2. Semantic Encoder + Classifier Head end-to-end 학습 (latent_dim ∈ {64,128,256})
python scripts/train_semantic.py --config configs/train_semantic.yaml

# 3. Semantic 평가
python scripts/eval_semantic.py --config configs/default.yaml

# 4. Baseline 평가 (Raw / JPEG)
python scripts/eval_baseline.py --config configs/default.yaml

# 5. 결과 시각화 (experiments/results/*.json → experiments/figures/*.png)
python scripts/plot_results.py
```

진행 보고용으로 epoch 수 / repeat 수를 축소한 [configs/quick_run.yaml](configs/quick_run.yaml) 이 별도로 존재한다.

**Device 주의:** [configs/default.yaml](configs/default.yaml) 와 quick_run의 `device` 필드 기본값은 `"mps"` (macOS) 다. Linux/GPU 환경에서는 `"cuda"` 또는 `"cpu"` 로 변경해야 한다.

테스트 프레임워크는 `pytest` 가 requirements에 포함되어 있으나 현재 `tests/` 디렉토리는 없다.

---

## 알려진 미완성 상태

스크립트들이 모두 `from src.data.cifar10 import get_loaders` 를 import 하지만 **`src/data/` 디렉토리는 아직 존재하지 않는다**. 학습/평가 스크립트를 실행하려면 먼저 `src/data/cifar10.py` 를 구현하고 `get_loaders(root, batch_size, num_workers)` 시그니처를 충족시켜야 한다. 같은 이유로 `notebooks/` 디렉토리도 아직 없다.

---

## 코드 아키텍처

데이터 흐름은 두 갈래로 나뉘며, 동일한 채널 시뮬레이터와 평가 지표를 공유한다:

```
[Image] ─ SemanticEncoder → Quantizer ─┐
                                       ├→ Channel → ClassifierHead → 예측
[Image] ─ Raw / JPEG 인코딩 ───────────┘                ResNet-18
```

**모듈 책임 분리:**
- [src/models/](src/models/) — `classifier.py` (ResNet-18), `semantic_encoder.py`, `semantic_decoder.py` (classifier head), `quantizer.py` (scalar 또는 K-means VQ).
- [src/channel/](src/channel/) — `base.py` 가 공통 인터페이스, 구현체로 `awgn.py` / `rayleigh.py` / `packet_loss.py`. 학습 루프 안에서도 forward 경로에 삽입된다 (예: `scripts/train_semantic.py` 가 `AWGNChannel` 을 직접 사용).
- [src/baseline/](src/baseline/) — `raw.py` (무손실 비트 전송), `jpeg.py` (JPEG 압축). 학습 없이 평가만 한다.
- [src/metrics/](src/metrics/) — `evaluate.py` 가 Top-1 / Accuracy-per-bit / bit-budget·SNR·packet-loss sweep 평가의 단일 진입점.
- [src/utils/](src/utils/) — `seed.py` (`set_seed`), `logger.py` (`get_logger`), `checkpoint.py` (`save_checkpoint`). 모든 학습/평가 스크립트의 첫 줄에서 `set_seed(cfg.seed)` 호출이 강제된다.

**설정 시스템:** `omegaconf` 기반. [configs/default.yaml](configs/default.yaml) 이 단일 진실의 원천이고, `train_classifier.yaml` / `train_semantic.yaml` / `quick_run.yaml` 이 이를 override 한다. CLI 의 `--config` 인자가 어떤 override 를 적용할지 결정한다.

**결과 저장:** 수치 결과는 `experiments/results/*.json` (CSV 아님 — `plot_results.py` 가 JSON 읽음), 그래프는 `experiments/figures/*.png`, 체크포인트는 `experiments/checkpoints/*.pth`. 모두 `.gitignore` 처리되어 레포에 포함되지 않는다.

---

## 핵심 설계 원칙

- **재현성:** 모든 학습/평가 스크립트는 `set_seed(cfg.seed)` 가 첫 동작. 기본 seed=42.
- **반복 실험:** 각 평가 조건은 `eval.repeat` (default 5) 회 반복, 평균 ± 표준편차 보고. quick_run 은 1회.
- **공통 sweep 축:**
  - Bit budget: 256B / 512B / 1KB / 5KB / 50KB
  - SNR: 0 / 5 / 10 / 20 dB
  - Packet loss: 1% / 5% / 10%
- **주지표:** Top-1 Accuracy. 효율성 비교에는 Accuracy per bit 사용.

---

## 보고서 (IEEE conference format)

- Progress Report: 진척 + 갱신된 Gantt chart.
- Final Report 섹션 구성: title / abstract / keywords / introduction / related works / methodology / performance evaluation / conclusions / references.
- 참고문헌 10개 이상, 본문에서 모두 최소 1회 인용.
