# 발표 슬라이드 설계 (8분 분량)

**과목**: 14897 Topics in Mobile Computing — Mini-Project Presentation
**발표자**: 이지혁 (20266243), 컴퓨터공학과 석사 1학기
**발표일**: 2026-05-27
**주제**: Semantic Communication vs. Raw-Bit Transmission

---

## 전체 흐름 (8분 = 480초)

| # | 슬라이드 | 시간 | 누적 | 핵심 메시지 |
|:-:|---|:-:|:-:|---|
| 1 | Title | 0:15 | 0:15 | 인사 + 주제 한 문장 |
| 2 | 동기 / Why Semantic? | 0:50 | 1:05 | Shannon → Semantic 패러다임 전환 |
| 3 | 시스템 구조 | 0:55 | 2:00 | Semantic vs JPEG 두 경로 다이어그램 |
| 4 | 방법론 | 1:00 | 3:00 | Encoder / Quantizer / Channel / Training |
| 5 | 결과 ① VQ 효율 | 1:00 | 4:00 | **0.75 B로 92.20%, JPEG 대비 1,700×** |
| 6 | 결과 ② 채널 강건성 | 0:50 | 4:50 | AWGN + Rayleigh sweep |
| 7 | 결과 ③ Channel-aware ablation | 1:10 | 6:00 | **저SNR +3%p 향상** |
| 8 | 결과 ④ 추정 전송 지연 | 0:45 | 6:45 | **LoRa 24.58 s → 6 ms** |
| 9 | 핵심 발견 + 한계 | 0:40 | 7:25 | 4 takeaways + limitations |
| 10 | Conclusion + Future Work | 0:25 | 7:50 | 한 줄 메시지 + 향후 |
| 11 | References | 0:10 | 8:00 | 핵심 ref 5개 (or backup) |

여유 10초 buffer.

---

## 슬라이드별 상세 설계

### Slide 1 — Title (0:15)

```
Semantic Communication vs. Raw-Bit Transmission
— CIFAR-10 분류 태스크에서의 성능 비교 —

이지혁 (20266243) · 컴퓨터공학과 석사 1학기
14897 Topics in Mobile Computing | 2026-05-27
```

**멘트**: "안녕하세요, 이지혁입니다. 오늘은 무선 채널 환경에서 semantic 통신이 JPEG 기반 원시 전송 대비 얼마나 효율적인지를 정량 비교한 결과를 발표하겠습니다."

---

### Slide 2 — 동기 (0:50)

**제목**: "왜 Semantic Communication인가?"

**좌측 다이어그램 (Shannon 패러다임):**

```
[Image] → [Source coder] → bits → [Channel] → bits → [Decoder] → [Image] → [Task]
                          모든 픽셀 복원 (~3 KB)
```

**우측 다이어그램 (Semantic 패러다임):**

```
[Image] → [Semantic encoder] → latent → [Channel] → [Task head] → [Task]
                          태스크 관련 정보만 (~0.75 B ~ 256 B)
```

**3 bullets:**
- 6G 시대: bits → tasks 직접 최적화 [1, 7]
- IoT/edge inference에서 대역폭·전력 제약
- **연구 문제**: 동일 분류 태스크에서 semantic이 JPEG 대비 얼마나 효율적인가?

**멘트**: "Shannon 시스템은 비트 무손실 복원이 목적이라, 분류만 필요한 상황에서도 픽셀 전체를 전송합니다. Semantic 통신은 태스크에 필요한 의미 정보만 보내자는 아이디어입니다."

---

### Slide 3 — 시스템 구조 (0:55)

**제목**: "시스템 구조"

**큰 다이어그램 (두 경로 비교):**

```
                              [AWGN / Rayleigh / Packet Loss]
                                         │
[Image] ─┬─ SemanticEncoder ─ Quantizer ─┼─ ClassifierHead ─ Pred   (Semantic)
         │   ResNet-18 backbone          │   2-layer MLP
         │   latent dim ∈ {64,128,256}   │
         │                               │
         └─ JPEG q ∈ {1..95} ────────────┼─ ResNet-18 (재학습 X) ─ Pred  (Baseline)
                                         │
```

**비교 축 (우측 박스):**
- 정확도 (Top-1)
- 효율 (Accuracy per bit)
- 강건성 (SNR / 패킷 손실)
- 전송 지연 (LoRa / LTE-M / 5G)

**멘트**: "두 경로를 동일 분류기 baseline 위에서 비교합니다. Semantic은 encoder가 latent를 만들고 양자화·채널·head로 이어집니다. Baseline은 JPEG으로 압축한 이미지를 무손실 학습된 ResNet-18에 그대로 입력합니다."

---

### Slide 4 — 방법론 (1:00)

**제목**: "Methodology"

**4-quadrant 레이아웃:**

**① Encoder + Head**
- ResNet-18 백본 + 선형 projection [3, 8]
- latent dim ∈ {64, 128, 256}
- end-to-end cross-entropy 학습

**② Quantization**
- **Scalar 8-bit uniform**: $d$ bytes 전송
- **Vector K-means** [5, 6]: ⌈log₂K⌉ bits, K ∈ {16, 64, 256, 1024}

**③ Channel models**
- AWGN: SNR ∈ {0, 5, 10, 20} dB
- Rayleigh: Perfect CSI 등화
- Block packet loss: 1% / 5% / 10%

**④ Channel-aware Training (ablation)**
- 매 batch마다 SNR ∈ [0, 20] dB random sampling [2, 6, 8]
- "다양한 채널 노출 = 강건한 표현"

**멘트**: "방법은 네 축입니다. Encoder는 ResNet-18 백본이고, 양자화는 scalar와 K-means VQ 두 가지를 비교합니다. 채널은 AWGN, Rayleigh, packet loss 세 종류고, 학습 시 채널 노출 여부도 ablation 했습니다."

---

### Slide 5 — 결과 ① VQ 효율 (1:00) 🔥

**제목**: "Vector Quantization is the Game Changer"

**좌측 그래프**: [accuracy_per_bit.png](../../experiments/figures/accuracy_per_bit.png) (Acc/bit log scale)

**우측 핵심 표:**

| 방식 | Bytes | Top-1 | Acc/Bit |
|---|---|---|---|
| JPEG q=95 | 1,300 B | 91.4% | 0.88×10⁻⁴ |
| Semantic Scalar dim=128 | 128 B | 92.3% | 9.0×10⁻⁴ |
| **Semantic VQ dim=128, K=64** | **0.75 B** | **92.2%** | **0.15** |

**큰 숫자 (slide 중앙 강조):**

> # **1,700× less bandwidth**
> # **at the same accuracy**

**멘트**: "이번 발표의 가장 강력한 결과입니다. K-means VQ로 latent 전체를 하나의 codebook 인덱스로 매핑하면 단 **6 bits**, 즉 0.75 byte 만으로 92.20% 정확도를 냅니다. JPEG가 동등 정확도를 내려면 1,300 바이트가 필요하니까, **약 1,700배** 적은 대역폭입니다."

---

### Slide 6 — 결과 ② 채널 강건성 (0:50)

**제목**: "Robustness under Realistic Channels"

**좌측**: [snr_robustness.png](../../experiments/figures/snr_robustness.png)
**우측**: [rayleigh_robustness.png](../../experiments/figures/rayleigh_robustness.png)

**핵심 bullets:**
- AWGN 0 dB에서도 dim=256은 **91.32%** 유지
- Rayleigh 0 dB: dim=256 **91.13%** (AWGN 대비 −0.2%p)
- 패킷 손실 10%에서도 dim=128 **92.37%** 유지

**멘트**: "AWGN뿐 아니라 Rayleigh 페이딩 환경에서도 강건합니다. AWGN 0 dB 대비 Rayleigh의 평균 정확도 손실은 0.3%p 미만입니다. 다만 std는 2-5배 커지는데, 이건 페이딩 계수의 무작위성 때문입니다."

---

### Slide 7 — 결과 ③ Channel-aware Ablation (1:10) 🔥

**제목**: "Channel-aware Training: Crucial at Low SNR"

**좌측 그래프**: [naive_vs_aware_awgn.png](../../experiments/figures/naive_vs_aware_awgn.png) (solid = naive, dashed = aware)
**우측 그래프**: [aware_improvement.png](../../experiments/figures/aware_improvement.png) (Δ bar chart)

**핵심 수치 (좌하):**
- AWGN 0 dB, dim=64: **+2.83%p**
- Rayleigh 0 dB, dim=64: **+3.03%p**

**우상 강조 박스:**

> ```
> Channel-aware dim=64 (64 B)
> > Channel-naive dim=128 (128 B)
>     in ALL SNRs.
> ```

**멘트**: "기존 학습은 채널 없이 진행했는데, 학계 표준은 random SNR sampling으로 학습하는 channel-aware training입니다. 효과는 저SNR에서 결정적입니다 — dim=64 모델 기준 AWGN 0 dB에서 +2.83%p, Rayleigh 0 dB에서 +3.03%p 향상됩니다. 더 흥미로운 건, **channel-aware dim=64가 channel-naive dim=128보다 모든 SNR에서 우수**합니다. 작은 latent도 다양한 채널 노출만 있으면 더 강건한 표현을 학습한다는 것을 보여줍니다."

---

### Slide 8 — 결과 ④ 추정 전송 지연 (0:45)

**제목**: "Latency: 4,100× Faster on IoT Links"

[latency_bar_lora.png](../../experiments/figures/latency_bar_lora.png) 또는 [latency_vs_bytes.png](../../experiments/figures/latency_vs_bytes.png) 중앙 배치

**핵심 표 (slide 하단):**

| Method | LoRa (1 kbps) | 5G (100 Mbps) |
|---|---|---|
| Raw | **24.58 s** | 246 µs |
| JPEG q=95 | 10.41 s | 104 µs |
| Semantic VQ | **6 ms** | **60 ns** |

**큰 숫자:**

> # **24.58 s → 6 ms** (LoRa, 4,100×)

**멘트**: "전송 지연을 LoRa, LTE-M, 5G 세 가지 link 가정으로 계산했습니다. LoRa 환경에서 raw는 24.58초가 걸리지만 semantic VQ는 6 ms로 약 4,100배 빠릅니다. 5G에서도 VQ는 60 나노초 수준으로, 네트워크 병목이 사실상 사라집니다."

---

### Slide 9 — 핵심 발견 + 한계 (0:40)

**제목**: "Takeaways & Limitations"

**좌측 (Takeaways):**
1. **VQ가 압도적 효율** — 6 bits로 92% 정확도, JPEG 대비 1,700×
2. **IoT 환경 latency 격차 극대** — LoRa 4,100× 빠름
3. **Channel-aware는 저SNR에서 필수** — +3%p, 작은 latent도 강건
4. **재현성 우수** — 5회 반복 std < 0.002

**우측 (Limitations):**
- Single-modality, single-task (이미지 분류)
- Rayleigh: Perfect CSI 가정 (imperfect CSI 미반영)
- VQ 평가: noiseless 가정 (인덱스 자체 채널 강건성 별도 분석 필요)

**멘트**: "정리하면 네 가지입니다 — VQ의 효율, IoT latency 격차, channel-aware의 저SNR 필수성, 그리고 결과 재현성. 한계는 단일 태스크 평가, perfect CSI 가정, 그리고 VQ 평가가 noiseless 가정이라는 점입니다."

---

### Slide 10 — Conclusion + Future Work (0:25)

**제목**: "Conclusion"

**중앙 메시지 (큰 글씨):**

> **"Task-oriented + Vector Quantization은**
> **무선 통신의 새로운 효율-강건성 frontier를 연다."**

**Future Work (작은 글씨):**
- Multi-task semantic communication [9]
- Channel-aware VQ — codebook–channel joint design [10]
- Modality 확장 (텍스트/음성) [8]
- Imperfect CSI / pilot overhead 반영
- Tiny-ImageNet, ImageNet으로 일반화

**멘트**: "결론은 task-oriented 접근과 vector quantization의 결합이 새로운 효율-강건성 frontier를 보여준다는 것이고, 향후 multi-task, channel-aware codebook, multimodal 확장이 자연스러운 다음 단계입니다. 감사합니다."

---

### Slide 11 — References (0:10, 또는 backup)

**화면에 5개 핵심 ref만 (글씨 작게, 나머지는 backup):**

- [1] Qin et al. "Semantic Communications: Principles and Challenges" 2021
- [2] Bourtsoulatze et al. "DeepJSCC" IEEE TCCN 2019
- [4] Shao et al. "Task-Oriented Communication / IB" IEEE JSAC 2022
- [6] Hu et al. "VQ-DeepSC" 2022
- [8] Xie et al. "DeepSC" IEEE TSP 2021

**(전체 11개 ref는 README와 final report에 명시.)**

---

## 발표 운영 팁

| 항목 | 권장 |
|---|---|
| **속도** | 슬라이드 5, 7이 핵심 — 평균보다 살짝 느리게, 큰 숫자에서 잠시 멈춤 |
| **시선** | 큰 숫자(1,700×, 4,100×, +3.03%p)는 한 번 더 짚어 강조 |
| **질문 대비** | Q1: "왜 channel-naive를 baseline으로?" → "초기 진행 보고서 시점 결과이고, ablation으로 비교 추가" / Q2: "VQ가 noiseless인 이유?" → 한계 인정 + future work |
| **데모** | 별도 데모 가능 시 슬라이드 8 뒤에 30초 분량 demo 영상 또는 라이브 (시간 8:30까지 늘어남) |
| **백업 슬라이드** | 전체 11개 ref, 자세한 5회 반복 std 표, hyperparameter 표, scalar_vs_vq.png |

## 시각 디자인 권장

- **색상 통일**: dim=64 빨강 (#E63946), dim=128 주황 (#F4A261), dim=256 청록 (#2A9D8F) — README/그래프와 동일
- **큰 숫자는 huge font** (60-80 pt): `1,700×`, `4,100×`, `+3.03%p`, `24.58 s → 6 ms`
- **각 슬라이드 하단**: 작은 페이지 번호 + "이지혁 | TMC Mini-Project | 2026-05-27"
- **공통 헤더**: 슬라이드 제목 좌측 정렬 + 우측 상단에 진행률 점 (●●○○○○○○○○○ 같은)

---

## 백업 슬라이드 후보

1. **자세한 결과 표** — AWGN/Rayleigh sweep 전체 (5회 반복 mean ± std)
2. **Hyperparameter 표** — optimizer, lr, epochs, batch size, seed
3. **Scalar vs Vector quantizer 그래프** — [scalar_vs_vq.png](../../experiments/figures/scalar_vs_vq.png)
4. **전체 11개 참고문헌**
5. **재현 가능성** — 깃 레포 URL, seed 고정, 실험 환경
6. **packet loss 강건성 결과** — [packet_loss_robustness.png](../../experiments/figures/packet_loss_robustness.png)
