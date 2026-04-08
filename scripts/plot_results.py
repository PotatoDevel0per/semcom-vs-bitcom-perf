"""실험 결과 시각화 스크립트 — Progress Report용 그래프 생성."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150
os.makedirs("experiments/figures", exist_ok=True)

# ── 데이터 로드 ──────────────────────────────────────────────
with open("experiments/results/baseline_eval.json") as f:
    baseline = json.load(f)

semantic_results = []
for dim in [64, 128, 256]:
    path = f"experiments/results/semantic_eval.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        semantic_results.extend([d for d in data if d["latent_dim"] == dim])

# ── Figure 1: Accuracy vs. Byte Budget ──────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Baseline: JPEG
jpeg = [r for r in baseline if r["method"] != "raw"]
jpeg_bytes = [r["byte_budget"] for r in jpeg]
jpeg_acc   = [r["top1_acc"] for r in jpeg]
raw_acc    = next(r["top1_acc"] for r in baseline if r["method"] == "raw")
raw_bytes  = next(r["byte_budget"] for r in baseline if r["method"] == "raw")

ax.plot(jpeg_bytes, jpeg_acc, 'o--', color='steelblue', label='JPEG (Baseline)', linewidth=1.5)
ax.axhline(raw_acc, color='gray', linestyle=':', linewidth=1.2, label=f'Raw (lossless, {raw_bytes}B)')

# Semantic
colors = {64: 'tomato', 128: 'darkorange', 256: 'seagreen'}
for dim in [64, 128, 256]:
    pts = [r for r in semantic_results if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10]
    if not pts:
        continue
    pts_sorted = sorted(pts, key=lambda r: r["avg_bytes"])
    ax.scatter([p["avg_bytes"] for p in pts_sorted],
               [p["top1_acc"] for p in pts_sorted],
               color=colors[dim], marker='*', s=120,
               label=f'Semantic dim={dim} (AWGN 10dB)')

ax.set_xlabel("Transmission Size (bytes)", fontsize=12)
ax.set_ylabel("Top-1 Accuracy", fontsize=12)
ax.set_title("Accuracy vs. Byte Budget", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig("experiments/figures/acc_vs_byte_budget.png")
plt.close()
print("Saved: acc_vs_byte_budget.png")

# ── Figure 2: Accuracy per Bit vs. Byte Budget ──────────────
fig, ax = plt.subplots(figsize=(7, 5))

jpeg_apb = [r["accuracy_per_bit"] for r in jpeg]
ax.plot(jpeg_bytes, [v * 1e4 for v in jpeg_apb], 'o--', color='steelblue', label='JPEG (Baseline)', linewidth=1.5)

for dim in [64, 128, 256]:
    pts = [r for r in semantic_results if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10]
    if not pts:
        continue
    pts_sorted = sorted(pts, key=lambda r: r["avg_bytes"])
    ax.scatter([p["avg_bytes"] for p in pts_sorted],
               [p["accuracy_per_bit"] * 1e4 for p in pts_sorted],
               color=colors[dim], marker='*', s=120,
               label=f'Semantic dim={dim} (AWGN 10dB)')

ax.set_xlabel("Transmission Size (bytes)", fontsize=12)
ax.set_ylabel("Accuracy per Bit (×10⁻⁴)", fontsize=12)
ax.set_title("Communication Efficiency: Accuracy per Bit", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig("experiments/figures/accuracy_per_bit.png")
plt.close()
print("Saved: accuracy_per_bit.png")

# ── Figure 3: SNR Robustness (AWGN) ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for dim in [64, 128, 256]:
    pts = [r for r in semantic_results if r["latent_dim"] == dim and r["channel"] == "awgn"]
    if not pts:
        continue
    pts_sorted = sorted(pts, key=lambda r: r["snr_db"])
    snrs = [p["snr_db"] for p in pts_sorted]
    accs = [p["top1_acc"] for p in pts_sorted]
    ax.plot(snrs, accs, 'o-', color=colors[dim], label=f'Semantic dim={dim}', linewidth=1.5, markersize=6)

ax.axhline(raw_acc, color='gray', linestyle=':', linewidth=1.2, label='Raw (lossless)')
ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("Top-1 Accuracy", fontsize=12)
ax.set_title("Robustness under AWGN Channel", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/figures/snr_robustness.png")
plt.close()
print("Saved: snr_robustness.png")

# ── Figure 4: Packet Loss Robustness ────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for dim in [64, 128, 256]:
    pts = [r for r in semantic_results if r["latent_dim"] == dim and r["channel"] == "packet_loss"]
    if not pts:
        continue
    pts_sorted = sorted(pts, key=lambda r: r["loss_rate"])
    loss_rates = [p["loss_rate"] * 100 for p in pts_sorted]
    accs = [p["top1_acc"] for p in pts_sorted]
    ax.plot(loss_rates, accs, 's-', color=colors[dim], label=f'Semantic dim={dim}', linewidth=1.5, markersize=6)

ax.set_xlabel("Packet Loss Rate (%)", fontsize=12)
ax.set_ylabel("Top-1 Accuracy", fontsize=12)
ax.set_title("Robustness under Packet Loss", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/figures/packet_loss_robustness.png")
plt.close()
print("Saved: packet_loss_robustness.png")

print("\nAll figures saved to experiments/figures/")
