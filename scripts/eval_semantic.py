"""Semantic 전송 방식 평가 — AWGN/Rayleigh/패킷손실 sweep + 5회 반복 + VQ 비교."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from omegaconf import OmegaConf

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.cifar10 import get_loaders
from src.models.semantic_encoder import SemanticEncoder
from src.models.semantic_decoder import SemanticClassifierHead
from src.models.quantizer import ScalarQuantizer, VectorQuantizer
from src.channel.base import BaseChannel
from src.channel.awgn import AWGNChannel
from src.channel.rayleigh import RayleighChannel
from src.channel.packet_loss import PacketLossChannel


class IdentityChannel(BaseChannel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def load_semantic(latent_dim, device, training="naive"):
    """training: 'naive' (기존) | 'aware' (channel-aware random SNR 학습)"""
    encoder = SemanticEncoder(latent_dim=latent_dim).to(device)
    head = SemanticClassifierHead(latent_dim=latent_dim).to(device)
    suffix = "_aware" if training == "aware" else ""
    ckpt = torch.load(
        f"experiments/checkpoints/semantic_enc{suffix}_dim{latent_dim}.pth",
        map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    encoder.eval(); head.eval()
    return encoder, head


def eval_scalar(encoder, head, loader, device, channel, quantizer):
    correct, total, total_bytes = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x).cpu()
            z_q, z_min, z_max = quantizer.quantize(z)
            total_bytes += quantizer.byte_count(z)
            z_deq = quantizer.dequantize(z_q, z_min, z_max).to(device)
            z_noisy = channel(z_deq)
            logits = head(z_noisy)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, total_bytes / total


def eval_vector(encoder, head, loader, device, vq):
    """Vector quantizer: latent 전체를 codebook 인덱스로 매핑 → 인덱스 전송."""
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x).cpu().numpy()
            idx = vq.quantize(z)
            z_rec = vq.dequantize(idx).astype(np.float32)
            z_rec = torch.from_numpy(z_rec).to(device)
            logits = head(z_rec)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    bytes_per_sample = vq.bits_per_vector() / 8.0
    return correct / total, bytes_per_sample


def repeated(fn, repeat, base_seed):
    accs, bytes_list = [], []
    for i in range(repeat):
        set_seed(base_seed + i)
        acc, b = fn()
        accs.append(acc); bytes_list.append(b)
    return {
        "top1_acc_mean": float(np.mean(accs)),
        "top1_acc_std": float(np.std(accs)),
        "avg_bytes": float(np.mean(bytes_list)),
        "repeat": repeat,
    }


def collect_train_latents(encoder, loader, device, max_samples=20000):
    """K-means fit용 train 잠재 벡터 수집 (속도/메모리 절충)."""
    feats, n = [], 0
    with torch.no_grad():
        for x, _ in loader:
            z = encoder(x.to(device)).cpu().numpy()
            feats.append(z); n += z.shape[0]
            if n >= max_samples: break
    return np.concatenate(feats, axis=0)[:max_samples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    logger = get_logger("eval_semantic")
    device = torch.device(cfg.device)

    train_loader, test_loader = get_loaders(
        cfg.data.root, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    scalar_q = ScalarQuantizer()
    repeat = int(cfg.eval.repeat)
    results = []

    for training in ["naive", "aware"]:
      for latent_dim in [64, 128, 256]:
        try:
            encoder, head = load_semantic(latent_dim, device, training=training)
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found for dim={latent_dim} training={training}; skip")
            continue
        logger.info(f"### Eval [{training}] dim={latent_dim} ###")

        # 1) AWGN SNR sweep
        for snr in cfg.eval.snr_sweep:
            stats = repeated(
                lambda snr=snr: eval_scalar(encoder, head, test_loader, device,
                                            AWGNChannel(snr_db=snr), scalar_q),
                repeat, cfg.seed)
            stats["accuracy_per_bit"] = stats["top1_acc_mean"] / (stats["avg_bytes"] * 8)
            r = {"method": "semantic", "training": training, "quantizer": "scalar",
                 "latent_dim": latent_dim, "channel": "awgn", "snr_db": snr, **stats}
            logger.info(f"[{training}] dim={latent_dim} AWGN SNR={snr}dB: "
                        f"{r['top1_acc_mean']:.4f}±{r['top1_acc_std']:.4f}")
            results.append(r)

        # 2) Rayleigh SNR sweep
        for snr in cfg.eval.snr_sweep:
            stats = repeated(
                lambda snr=snr: eval_scalar(encoder, head, test_loader, device,
                                            RayleighChannel(snr_db=snr), scalar_q),
                repeat, cfg.seed)
            stats["accuracy_per_bit"] = stats["top1_acc_mean"] / (stats["avg_bytes"] * 8)
            r = {"method": "semantic", "training": training, "quantizer": "scalar",
                 "latent_dim": latent_dim, "channel": "rayleigh", "snr_db": snr, **stats}
            logger.info(f"[{training}] dim={latent_dim} Rayleigh SNR={snr}dB: "
                        f"{r['top1_acc_mean']:.4f}±{r['top1_acc_std']:.4f}")
            results.append(r)

        # 3) Packet loss sweep
        for loss_rate in cfg.eval.packet_loss_sweep:
            stats = repeated(
                lambda lr=loss_rate: eval_scalar(encoder, head, test_loader, device,
                                                 PacketLossChannel(loss_rate=lr), scalar_q),
                repeat, cfg.seed)
            stats["accuracy_per_bit"] = stats["top1_acc_mean"] / (stats["avg_bytes"] * 8)
            r = {"method": "semantic", "training": training, "quantizer": "scalar",
                 "latent_dim": latent_dim, "channel": "packet_loss",
                 "loss_rate": loss_rate, **stats}
            logger.info(f"[{training}] dim={latent_dim} PacketLoss={loss_rate}: "
                        f"{r['top1_acc_mean']:.4f}±{r['top1_acc_std']:.4f}")
            results.append(r)

        # 4) Vector quantizer 비교 (noiseless)
        logger.info(f"=> [{training}] K-means VQ fit (dim={latent_dim})")
        train_z = collect_train_latents(encoder, train_loader, device)
        for K in [16, 64, 256, 1024]:
            vq = VectorQuantizer(num_codes=K)
            vq.fit(train_z)
            acc, b = eval_vector(encoder, head, test_loader, device, vq)
            r = {"method": "semantic", "training": training, "quantizer": "vector",
                 "latent_dim": latent_dim, "channel": "noiseless",
                 "num_codes": K, "top1_acc_mean": acc, "top1_acc_std": 0.0,
                 "avg_bytes": b, "repeat": 1,
                 "accuracy_per_bit": acc / (b * 8 + 1e-9)}
            logger.info(f"dim={latent_dim} VQ K={K}: acc={acc:.4f}, "
                        f"bytes={b:.3f}, acc/bit={r['accuracy_per_bit']:.4f}")
            results.append(r)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/semantic_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to experiments/results/semantic_eval.json")
