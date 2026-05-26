"""실험 결과 시각화 — AWGN/Rayleigh/패킷손실 sweep + Scalar/VQ 비교."""
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

with open("experiments/results/baseline_eval.json") as f:
    baseline = json.load(f)
with open("experiments/results/semantic_eval.json") as f:
    semantic = json.load(f)

scalar_sem = [r for r in semantic if r["quantizer"] == "scalar"]
vector_sem = [r for r in semantic if r["quantizer"] == "vector"]

raw = next(r for r in baseline if r["method"] == "raw")
jpeg = sorted([r for r in baseline if r["method"] != "raw"], key=lambda r: r["byte_budget"])
colors = {64: 'tomato', 128: 'darkorange', 256: 'seagreen'}

# ── Figure 1: Accuracy vs. Byte Budget (AWGN 10dB) ─────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([r["byte_budget"] for r in jpeg], [r["top1_acc"] for r in jpeg],
        'o--', color='steelblue', label='JPEG (Baseline)', linewidth=1.5)
ax.axhline(raw["top1_acc"], color='gray', linestyle=':', linewidth=1.2,
           label=f'Raw lossless ({raw["byte_budget"]}B)')
for dim in [64, 128, 256]:
    pts = [r for r in scalar_sem
           if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10]
    if not pts: continue
    ax.errorbar([p["avg_bytes"] for p in pts], [p["top1_acc_mean"] for p in pts],
                yerr=[p["top1_acc_std"] for p in pts], fmt='*',
                color=colors[dim], markersize=14, capsize=4,
                label=f'Semantic dim={dim} (AWGN 10dB)')
ax.set_xlabel("Transmission Size (bytes)"); ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Accuracy vs. Byte Budget")
ax.set_xscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("experiments/figures/acc_vs_byte_budget.png"); plt.close()
print("Saved: acc_vs_byte_budget.png")

# ── Figure 2: Accuracy per Bit vs. Byte Budget ────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([r["byte_budget"] for r in jpeg], [r["accuracy_per_bit"] * 1e4 for r in jpeg],
        'o--', color='steelblue', label='JPEG (Baseline)', linewidth=1.5)
for dim in [64, 128, 256]:
    pts = [r for r in scalar_sem
           if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10]
    if not pts: continue
    ax.scatter([p["avg_bytes"] for p in pts],
               [p["accuracy_per_bit"] * 1e4 for p in pts],
               color=colors[dim], marker='*', s=180,
               label=f'Semantic dim={dim} (AWGN 10dB)')
ax.set_xlabel("Transmission Size (bytes)"); ax.set_ylabel("Accuracy per Bit (×10⁻⁴)")
ax.set_title("Communication Efficiency: Accuracy per Bit")
ax.set_xscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("experiments/figures/accuracy_per_bit.png"); plt.close()
print("Saved: accuracy_per_bit.png")

# ── Figure 3: AWGN SNR robustness ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for dim in [64, 128, 256]:
    pts = sorted([r for r in scalar_sem if r["latent_dim"] == dim and r["channel"] == "awgn"],
                 key=lambda r: r["snr_db"])
    if not pts: continue
    ax.errorbar([p["snr_db"] for p in pts], [p["top1_acc_mean"] for p in pts],
                yerr=[p["top1_acc_std"] for p in pts], fmt='o-',
                color=colors[dim], linewidth=1.5, markersize=6, capsize=3,
                label=f'Semantic dim={dim}')
ax.axhline(raw["top1_acc"], color='gray', linestyle=':', linewidth=1.2, label='Raw (lossless)')
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Robustness under AWGN Channel")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("experiments/figures/snr_robustness.png"); plt.close()
print("Saved: snr_robustness.png")

# ── Figure 4: Rayleigh SNR robustness ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for dim in [64, 128, 256]:
    pts = sorted([r for r in scalar_sem if r["latent_dim"] == dim and r["channel"] == "rayleigh"],
                 key=lambda r: r["snr_db"])
    if not pts: continue
    ax.errorbar([p["snr_db"] for p in pts], [p["top1_acc_mean"] for p in pts],
                yerr=[p["top1_acc_std"] for p in pts], fmt='s-',
                color=colors[dim], linewidth=1.5, markersize=6, capsize=3,
                label=f'Semantic dim={dim}')
ax.axhline(raw["top1_acc"], color='gray', linestyle=':', linewidth=1.2, label='Raw (lossless)')
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Robustness under Rayleigh Fading Channel")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("experiments/figures/rayleigh_robustness.png"); plt.close()
print("Saved: rayleigh_robustness.png")

# ── Figure 5: Packet loss robustness ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for dim in [64, 128, 256]:
    pts = sorted([r for r in scalar_sem if r["latent_dim"] == dim and r["channel"] == "packet_loss"],
                 key=lambda r: r["loss_rate"])
    if not pts: continue
    ax.errorbar([p["loss_rate"] * 100 for p in pts], [p["top1_acc_mean"] for p in pts],
                yerr=[p["top1_acc_std"] for p in pts], fmt='s-',
                color=colors[dim], linewidth=1.5, markersize=6, capsize=3,
                label=f'Semantic dim={dim}')
ax.set_xlabel("Packet Loss Rate (%)"); ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Robustness under Block Packet Loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("experiments/figures/packet_loss_robustness.png"); plt.close()
print("Saved: packet_loss_robustness.png")

# ── Figure 6: Scalar vs Vector quantizer ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for dim in [64, 128, 256]:
    sc = [r for r in scalar_sem if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 20]
    if sc:
        ax.scatter([p["avg_bytes"] for p in sc], [p["top1_acc_mean"] for p in sc],
                   color=colors[dim], marker='*', s=180,
                   label=f'Scalar dim={dim}')
    vq = sorted([r for r in vector_sem if r["latent_dim"] == dim], key=lambda r: r["avg_bytes"])
    if vq:
        ax.plot([p["avg_bytes"] for p in vq], [p["top1_acc_mean"] for p in vq],
                'o-', color=colors[dim], linewidth=1.2, markersize=6, alpha=0.7,
                label=f'VQ dim={dim} (K∈{{16,64,256,1024}})')
ax.set_xlabel("Transmission Size (bytes)"); ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Scalar (8-bit) vs. Vector Quantization (K-means)")
ax.set_xscale('log'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("experiments/figures/scalar_vs_vq.png"); plt.close()
print("Saved: scalar_vs_vq.png")

print("\nAll figures saved to experiments/figures/")
