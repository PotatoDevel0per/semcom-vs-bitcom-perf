"""실험 결과 시각화 — AWGN/Rayleigh/패킷손실 sweep + Scalar/VQ 비교."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
os.makedirs("experiments/figures", exist_ok=True)


def bytes_fmt(x, _=None):
    """0.5B / 64B / 1KB / 3KB 형식으로 byte 값 포맷."""
    if x < 1:
        return f"{x:g} B"
    if x < 1000:
        return f"{int(round(x))} B"
    return f"{x/1024:.1f} KB"


def setup_byte_xaxis(ax, ticks):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(bytes_fmt))
    plt.setp(ax.get_xticklabels(), rotation=0)


with open("experiments/results/baseline_eval.json") as f:
    baseline = json.load(f)
with open("experiments/results/semantic_eval.json") as f:
    semantic = json.load(f)

scalar_sem = [r for r in semantic if r["quantizer"] == "scalar"]
vector_sem = [r for r in semantic if r["quantizer"] == "vector"]
# 기존 결과에 training 필드가 없으면 'naive'로 가정 (하위 호환)
for r in semantic:
    r.setdefault("training", "naive")

scalar_naive = [r for r in scalar_sem if r.get("training", "naive") == "naive"]
scalar_aware = [r for r in scalar_sem if r.get("training") == "aware"]
vector_naive = [r for r in vector_sem if r.get("training", "naive") == "naive"]
vector_aware = [r for r in vector_sem if r.get("training") == "aware"]
has_aware = len(scalar_aware) > 0

raw = next(r for r in baseline if r["method"] == "raw")
jpeg = sorted([r for r in baseline if r["method"] != "raw"], key=lambda r: r["byte_budget"])
colors = {64: '#E63946', 128: '#F4A261', 256: '#2A9D8F'}

# ── Figure 1: Accuracy vs. Byte Budget (AWGN 10dB) ────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot([r["byte_budget"] for r in jpeg], [r["top1_acc"] * 100 for r in jpeg],
        'o--', color='#264653', label='JPEG (Baseline)', linewidth=1.8, markersize=8)
ax.axhline(raw["top1_acc"] * 100, color='gray', linestyle=':', linewidth=1.5,
           label=f'Raw lossless ({bytes_fmt(raw["byte_budget"])}, {raw["top1_acc"]*100:.1f}%)')

for dim in [64, 128, 256]:
    pts = [r for r in scalar_naive
           if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10]
    if not pts: continue
    p = pts[0]
    ax.errorbar([p["avg_bytes"]], [p["top1_acc_mean"] * 100],
                yerr=[p["top1_acc_std"] * 100], fmt='*',
                color=colors[dim], markersize=22, capsize=5, markeredgecolor='black',
                markeredgewidth=0.8, label=f'Semantic dim={dim} (AWGN 10 dB)')
    ax.annotate(f"{p['top1_acc_mean']*100:.1f}%",
                xy=(p["avg_bytes"], p["top1_acc_mean"]*100),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color=colors[dim], fontweight='bold')

ax.set_xlabel("Transmission Size")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Accuracy vs. Byte Budget")
setup_byte_xaxis(ax, [64, 128, 256, 700, 1024, 3072])
ax.set_ylim(15, 100)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.35, which='major')
plt.tight_layout()
plt.savefig("experiments/figures/acc_vs_byte_budget.png", bbox_inches='tight')
plt.close()
print("Saved: acc_vs_byte_budget.png")

# ── Figure 2: Accuracy per Bit vs. Byte Budget ────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot([r["byte_budget"] for r in jpeg], [r["accuracy_per_bit"] * 1e4 for r in jpeg],
        'o--', color='#264653', label='JPEG (Baseline)', linewidth=1.8, markersize=8)
for dim in [64, 128, 256]:
    pts = [r for r in scalar_naive
           if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10]
    if not pts: continue
    p = pts[0]
    ax.scatter([p["avg_bytes"]], [p["accuracy_per_bit"] * 1e4],
               color=colors[dim], marker='*', s=400, edgecolor='black', linewidth=0.8,
               label=f'Semantic dim={dim} (AWGN 10 dB)', zorder=5)
    ax.annotate(f"{p['accuracy_per_bit']*1e4:.1f}",
                xy=(p["avg_bytes"], p["accuracy_per_bit"]*1e4),
                xytext=(8, -3), textcoords='offset points',
                fontsize=9, color=colors[dim], fontweight='bold')

ax.set_xlabel("Transmission Size")
ax.set_ylabel("Accuracy per Bit (×10⁻⁴)")
ax.set_title("Communication Efficiency: Accuracy per Bit")
setup_byte_xaxis(ax, [64, 128, 256, 700, 1024, 3072])
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.35, which='major')
plt.tight_layout()
plt.savefig("experiments/figures/accuracy_per_bit.png", bbox_inches='tight')
plt.close()
print("Saved: accuracy_per_bit.png")

# ── Figure 3: AWGN SNR robustness ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for dim in [64, 128, 256]:
    pts = sorted([r for r in scalar_naive if r["latent_dim"] == dim and r["channel"] == "awgn"],
                 key=lambda r: r["snr_db"])
    if not pts: continue
    ax.errorbar([p["snr_db"] for p in pts], [p["top1_acc_mean"] * 100 for p in pts],
                yerr=[p["top1_acc_std"] * 100 for p in pts], fmt='o-',
                color=colors[dim], linewidth=2.0, markersize=8, capsize=4,
                label=f'Semantic dim={dim}')
ax.axhline(raw["top1_acc"] * 100, color='gray', linestyle=':', linewidth=1.5,
           label=f'Raw lossless ({raw["top1_acc"]*100:.1f}%)')
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Robustness under AWGN Channel")
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_ylim(86, 95)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("experiments/figures/snr_robustness.png", bbox_inches='tight')
plt.close()
print("Saved: snr_robustness.png")

# ── Figure 4: Rayleigh SNR robustness ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for dim in [64, 128, 256]:
    pts = sorted([r for r in scalar_naive if r["latent_dim"] == dim and r["channel"] == "rayleigh"],
                 key=lambda r: r["snr_db"])
    if not pts: continue
    ax.errorbar([p["snr_db"] for p in pts], [p["top1_acc_mean"] * 100 for p in pts],
                yerr=[p["top1_acc_std"] * 100 for p in pts], fmt='s-',
                color=colors[dim], linewidth=2.0, markersize=8, capsize=4,
                label=f'Semantic dim={dim}')
ax.axhline(raw["top1_acc"] * 100, color='gray', linestyle=':', linewidth=1.5,
           label=f'Raw lossless ({raw["top1_acc"]*100:.1f}%)')
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Robustness under Rayleigh Fading Channel (Perfect CSI)")
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_ylim(86, 95)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("experiments/figures/rayleigh_robustness.png", bbox_inches='tight')
plt.close()
print("Saved: rayleigh_robustness.png")

# ── Figure 5: Packet loss robustness ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for dim in [64, 128, 256]:
    pts = sorted([r for r in scalar_naive if r["latent_dim"] == dim and r["channel"] == "packet_loss"],
                 key=lambda r: r["loss_rate"])
    if not pts: continue
    ax.errorbar([p["loss_rate"] * 100 for p in pts], [p["top1_acc_mean"] * 100 for p in pts],
                yerr=[p["top1_acc_std"] * 100 for p in pts], fmt='D-',
                color=colors[dim], linewidth=2.0, markersize=8, capsize=4,
                label=f'Semantic dim={dim}')
ax.set_xlabel("Packet Loss Rate (%)")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Robustness under Block Packet Loss")
ax.set_xticks([1, 5, 10])
ax.set_ylim(91, 93.5)
ax.legend(loc='lower left', framealpha=0.95)
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("experiments/figures/packet_loss_robustness.png", bbox_inches='tight')
plt.close()
print("Saved: packet_loss_robustness.png")

# ── Figure 6: Scalar vs Vector quantizer ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
for dim in [64, 128, 256]:
    sc = [r for r in scalar_naive
          if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 20]
    if sc:
        p = sc[0]
        ax.scatter([p["avg_bytes"]], [p["top1_acc_mean"] * 100],
                   color=colors[dim], marker='*', s=400, edgecolor='black', linewidth=0.8,
                   zorder=5, label=f'Scalar 8-bit, dim={dim} ({bytes_fmt(p["avg_bytes"])})')
    vq = sorted([r for r in vector_naive if r["latent_dim"] == dim], key=lambda r: r["avg_bytes"])
    if vq:
        xs = [p["avg_bytes"] for p in vq]
        ys = [p["top1_acc_mean"] * 100 for p in vq]
        ax.plot(xs, ys, 'o-', color=colors[dim], linewidth=1.8, markersize=8, alpha=0.85,
                label=f'K-means VQ, dim={dim}')
        for p in vq:
            ax.annotate(f"K={p['num_codes']}",
                        xy=(p["avg_bytes"], p["top1_acc_mean"]*100),
                        xytext=(0, -14), textcoords='offset points',
                        fontsize=8, color=colors[dim], ha='center')
ax.set_xlabel("Transmission Size")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Scalar (8-bit) vs. Vector Quantization (K-means)")
setup_byte_xaxis(ax, [0.5, 0.75, 1, 1.25, 64, 128, 256])
ax.set_ylim(88, 94)
ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
ax.grid(True, alpha=0.35, which='major')
plt.tight_layout()
plt.savefig("experiments/figures/scalar_vs_vq.png", bbox_inches='tight')
plt.close()
print("Saved: scalar_vs_vq.png")

# ── Figure 7 & 8: Channel-naive vs Channel-aware (AWGN / Rayleigh) ────
if has_aware:
    for channel_name, fname, title, marker_n, marker_a in [
        ("awgn", "naive_vs_aware_awgn.png",
         "Channel-naive vs Channel-aware Training (AWGN)", 'o', '^'),
        ("rayleigh", "naive_vs_aware_rayleigh.png",
         "Channel-naive vs Channel-aware Training (Rayleigh)", 's', 'D'),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for dim in [64, 128, 256]:
            naive_pts = sorted([r for r in scalar_naive
                                if r["latent_dim"] == dim and r["channel"] == channel_name],
                               key=lambda r: r["snr_db"])
            aware_pts = sorted([r for r in scalar_aware
                                if r["latent_dim"] == dim and r["channel"] == channel_name],
                               key=lambda r: r["snr_db"])
            if naive_pts:
                ax.errorbar([p["snr_db"] for p in naive_pts],
                            [p["top1_acc_mean"] * 100 for p in naive_pts],
                            yerr=[p["top1_acc_std"] * 100 for p in naive_pts],
                            fmt=f'{marker_n}-', color=colors[dim], linewidth=1.8,
                            markersize=7, capsize=3, alpha=0.85,
                            label=f'dim={dim}, naive')
            if aware_pts:
                ax.errorbar([p["snr_db"] for p in aware_pts],
                            [p["top1_acc_mean"] * 100 for p in aware_pts],
                            yerr=[p["top1_acc_std"] * 100 for p in aware_pts],
                            fmt=f'{marker_a}--', color=colors[dim], linewidth=1.8,
                            markersize=8, capsize=3, markerfacecolor='white',
                            markeredgewidth=1.5, label=f'dim={dim}, aware')
        ax.axhline(raw["top1_acc"] * 100, color='gray', linestyle=':', linewidth=1.3,
                   label=f'Raw lossless ({raw["top1_acc"]*100:.1f}%)')
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(title)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(86, 95)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=9, ncol=2)
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(f"experiments/figures/{fname}", bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

    # ── Figure 9: Accuracy improvement Δ (aware − naive) ─────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.25
    snrs = sorted({r["snr_db"] for r in scalar_aware if r["channel"] == "awgn"})
    x_idx = np.arange(len(snrs))
    for i, dim in enumerate([64, 128, 256]):
        deltas = []
        for snr in snrs:
            n = next((r for r in scalar_naive
                      if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == snr), None)
            a = next((r for r in scalar_aware
                      if r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == snr), None)
            if n and a:
                deltas.append((a["top1_acc_mean"] - n["top1_acc_mean"]) * 100)
            else:
                deltas.append(0)
        bars = ax.bar(x_idx + (i - 1) * width, deltas, width,
                      color=colors[dim], edgecolor='black', linewidth=0.5,
                      label=f'dim={dim}')
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    d + (0.05 if d >= 0 else -0.15),
                    f"{d:+.2f}", ha='center', fontsize=8,
                    color=colors[dim], fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{s} dB" for s in snrs])
    ax.set_xlabel("AWGN SNR")
    ax.set_ylabel("Accuracy Δ (aware − naive, %p)")
    ax.set_title("Channel-aware Training Improvement under AWGN")
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.35)
    plt.tight_layout()
    plt.savefig("experiments/figures/aware_improvement.png", bbox_inches='tight')
    plt.close()
    print("Saved: aware_improvement.png")

print("\nAll figures saved to experiments/figures/")
