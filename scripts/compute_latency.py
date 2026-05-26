"""추정 전송 지연 계산 — 가정된 전송률 기반.

각 전송 방식의 평균 페이로드 크기에 가정된 전송률을 적용해 전송 지연을 계산하고,
결과를 JSON / 표 / 그래프로 출력한다.
"""
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

# ── 가정된 전송률 ────────────────────────────────────────────────────
# (이름, bits per second)
LINKS = [
    ("LoRa (IoT)",      1_000),          # 1 kbps
    ("LTE-M",           1_000_000),      # 1 Mbps
    ("5G mid-band",     100_000_000),    # 100 Mbps
]


def latency_ms(byte_count: float, bps: float) -> float:
    return byte_count * 8.0 / bps * 1000.0  # milliseconds


def fmt_time(ms: float) -> str:
    if ms < 1e-3:
        return f"{ms*1e3:.2f} µs"
    if ms < 1:
        return f"{ms*1e3:.1f} µs"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms/1000:.2f} s"


def bytes_fmt(x, _=None):
    if x < 1:
        return f"{x:g} B"
    if x < 1000:
        return f"{int(round(x))} B"
    return f"{x/1024:.1f} KB"


if __name__ == "__main__":
    with open("experiments/results/baseline_eval.json") as f:
        baseline = json.load(f)
    with open("experiments/results/semantic_eval.json") as f:
        semantic = json.load(f)

    # ── 대표 전송 방식 선택 ─────────────────────────────────────────
    raw = next(r for r in baseline if r["method"] == "raw")
    methods = [
        ("Raw (lossless)",              raw["byte_budget"],     raw["top1_acc"]),
        ("JPEG q=60",                   853.8,                  0.7949),
        ("JPEG q=95",                   1300.8,                 0.9142),
    ]
    for dim in [64, 128, 256]:
        sc = next((r for r in semantic if r["quantizer"] == "scalar" and
                   r["latent_dim"] == dim and r["channel"] == "awgn" and r["snr_db"] == 10),
                  None)
        if sc:
            methods.append((f"Semantic Scalar dim={dim} (AWGN 10dB)",
                            sc["avg_bytes"], sc["top1_acc_mean"]))
    for dim in [64, 128, 256]:
        vq_best = max((r for r in semantic if r["quantizer"] == "vector" and
                       r["latent_dim"] == dim),
                      key=lambda r: r["top1_acc_mean"], default=None)
        if vq_best:
            methods.append((f"Semantic VQ dim={dim} K={vq_best['num_codes']}",
                            vq_best["avg_bytes"], vq_best["top1_acc_mean"]))

    # ── 표 출력 + JSON 저장 ────────────────────────────────────────
    results = []
    print(f"{'Method':<42} {'Bytes':>10} {'Top-1':>8}  " +
          "  ".join(f"{name:>14}" for name, _ in LINKS))
    print("-" * (42 + 10 + 9 + 16 * len(LINKS)))
    for name, byte_cnt, acc in methods:
        row = {"method": name, "bytes": byte_cnt, "top1_acc": acc, "latency_ms": {}}
        line = f"{name:<42} {byte_cnt:>10.2f} {acc*100:>7.2f}%  "
        for link_name, bps in LINKS:
            lat = latency_ms(byte_cnt, bps)
            row["latency_ms"][link_name] = lat
            line += f"  {fmt_time(lat):>14}"
        print(line)
        results.append(row)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/latency_estimates.json", "w") as f:
        json.dump({"links_bps": dict(LINKS), "methods": results}, f, indent=2)
    print("\nSaved: experiments/results/latency_estimates.json")

    # ── 그래프 1: Bytes vs Latency (3개 link, log-log) ─────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    byte_range = np.logspace(np.log10(0.5), np.log10(3072), 200)
    link_colors = {"LoRa (IoT)": "#E63946", "LTE-M": "#F4A261", "5G mid-band": "#2A9D8F"}
    for link_name, bps in LINKS:
        ax.plot(byte_range, [latency_ms(b, bps) for b in byte_range],
                '-', color=link_colors[link_name], linewidth=2.0, label=link_name)

    # 대표 방식 표시
    marker_for = {"Raw": ('s', '#264653'), "JPEG": ('o', '#264653'),
                  "Scalar": ('*', '#E63946'), "VQ": ('D', '#2A9D8F')}
    for row in results:
        name = row["method"]
        if "Raw" in name: key = "Raw"
        elif "JPEG" in name: key = "JPEG"
        elif "Scalar" in name: key = "Scalar"
        else: key = "VQ"
        marker, _ = marker_for[key]
        # LTE-M (중간 link) 기준 점 표시
        lat = latency_ms(row["bytes"], 1_000_000)
        ax.scatter([row["bytes"]], [lat], marker=marker, s=70,
                   edgecolor='black', linewidth=0.6, zorder=5,
                   color={'Raw': '#264653', 'JPEG': '#88B0B8',
                          'Scalar': '#E63946', 'VQ': '#2A9D8F'}[key])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Transmission Size")
    ax.set_ylabel("Estimated Latency (ms)")
    ax.set_title("Estimated Transmission Latency vs. Payload Size")
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 10, 100, 1024, 3072]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(bytes_fmt))
    ax.legend(loc='upper left', framealpha=0.95, title='Link assumption')
    ax.grid(True, alpha=0.35, which='major')
    ax.grid(True, alpha=0.15, which='minor')
    os.makedirs("experiments/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("experiments/figures/latency_vs_bytes.png", bbox_inches='tight')
    plt.close()
    print("Saved: experiments/figures/latency_vs_bytes.png")

    # ── 그래프 2: Method 별 latency bar (LoRa 기준 — 차이 가장 극명) ─
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [r["method"].replace(" (AWGN 10dB)", "") for r in results]
    lats = [r["latency_ms"]["LoRa (IoT)"] for r in results]
    accs = [r["top1_acc"] * 100 for r in results]
    color_for_bar = []
    for n in names:
        if "Raw" in n: color_for_bar.append('#264653')
        elif "JPEG" in n: color_for_bar.append('#88B0B8')
        elif "Scalar" in n: color_for_bar.append('#E63946')
        else: color_for_bar.append('#2A9D8F')
    bars = ax.barh(range(len(names)), lats, color=color_for_bar, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel("Estimated Latency on LoRa (1 kbps) [log scale, ms]")
    ax.set_title("Estimated Latency per Method (LoRa IoT Link, 1 kbps)")
    for i, (bar, lat, acc) in enumerate(zip(bars, lats, accs)):
        ax.text(lat * 1.05, i, f"{fmt_time(lat)}  ({acc:.1f}%)",
                va='center', fontsize=9)
    ax.grid(True, axis='x', alpha=0.35, which='major')
    plt.tight_layout()
    plt.savefig("experiments/figures/latency_bar_lora.png", bbox_inches='tight')
    plt.close()
    print("Saved: experiments/figures/latency_bar_lora.png")
