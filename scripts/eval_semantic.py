"""Semantic 전송 방식 평가 스크립트 — 비트 예산 및 채널 조건 스윕."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from omegaconf import OmegaConf

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.cifar10 import get_loaders
from src.models.semantic_encoder import SemanticEncoder
from src.models.semantic_decoder import SemanticClassifierHead
from src.models.quantizer import ScalarQuantizer
from src.channel.awgn import AWGNChannel
from src.channel.rayleigh import RayleighChannel
from src.channel.packet_loss import PacketLossChannel


def load_semantic(latent_dim, device):
    encoder = SemanticEncoder(latent_dim=latent_dim).to(device)
    head = SemanticClassifierHead(latent_dim=latent_dim).to(device)
    ckpt = torch.load(f"experiments/checkpoints/semantic_enc_dim{latent_dim}.pth", map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    encoder.eval(); head.eval()
    return encoder, head


def run_eval(encoder, head, loader, device, channel, quantizer):
    correct, total, total_bytes = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x)
            z_cpu = z.cpu()
            z_q, z_min, z_max = quantizer.quantize(z_cpu)
            total_bytes += quantizer.byte_count(z_cpu)
            z_deq = quantizer.dequantize(z_q, z_min, z_max).to(device)
            z_noisy = channel(z_deq)
            logits = head(z_noisy)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    avg_bytes = total_bytes / total
    return acc, avg_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    logger = get_logger("eval_semantic")
    device = torch.device(cfg.device)

    _, test_loader = get_loaders(cfg.data.root, batch_size=cfg.data.batch_size,
                                  num_workers=cfg.data.num_workers)
    quantizer = ScalarQuantizer()
    results = []

    # 1) latent_dim 별 SNR sweep (AWGN)
    for latent_dim in [64, 128, 256]:
        try:
            encoder, head = load_semantic(latent_dim, device)
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found for latent_dim={latent_dim}, skipping.")
            continue

        for snr in cfg.eval.snr_sweep:
            channel = AWGNChannel(snr_db=snr)
            acc, avg_bytes = run_eval(encoder, head, test_loader, device, channel, quantizer)
            r = {"method": "semantic", "latent_dim": latent_dim, "channel": "awgn",
                 "snr_db": snr, "top1_acc": acc, "avg_bytes": avg_bytes,
                 "accuracy_per_bit": acc / (avg_bytes * 8)}
            logger.info(f"Semantic dim={latent_dim} AWGN SNR={snr}dB: acc={acc:.4f}, bytes={avg_bytes:.1f}")
            results.append(r)

        # 2) 패킷 손실 sweep
        for loss_rate in cfg.eval.packet_loss_sweep:
            channel = PacketLossChannel(loss_rate=loss_rate)
            acc, avg_bytes = run_eval(encoder, head, test_loader, device, channel, quantizer)
            r = {"method": "semantic", "latent_dim": latent_dim, "channel": "packet_loss",
                 "loss_rate": loss_rate, "top1_acc": acc, "avg_bytes": avg_bytes,
                 "accuracy_per_bit": acc / (avg_bytes * 8)}
            logger.info(f"Semantic dim={latent_dim} PacketLoss={loss_rate}: acc={acc:.4f}")
            results.append(r)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/semantic_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to experiments/results/semantic_eval.json")
