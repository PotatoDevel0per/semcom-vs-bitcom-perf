"""Semantic Encoder + Classifier Head end-to-end 학습 스크립트."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.checkpoint import save_checkpoint
from src.data.cifar10 import get_loaders
from src.models.semantic_encoder import SemanticEncoder
from src.models.semantic_decoder import SemanticClassifierHead
from src.models.quantizer import ScalarQuantizer
from src.channel.awgn import AWGNChannel


def train(cfg):
    set_seed(cfg.seed)
    logger = get_logger(f"train_semantic_dim{cfg.model.latent_dim}")
    device = torch.device(cfg.device)

    train_loader, test_loader = get_loaders(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    encoder = SemanticEncoder(latent_dim=cfg.model.latent_dim).to(device)
    head = SemanticClassifierHead(latent_dim=cfg.model.latent_dim).to(device)
    channel = AWGNChannel(snr_db=cfg.channel.snr_db)
    quantizer = ScalarQuantizer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    best_acc = 0.0
    results = []

    for epoch in range(1, cfg.train.epochs + 1):
        # Train (채널 없이 학습 — 양자화/채널은 평가 시 적용)
        encoder.train(); head.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = encoder(x)
            logits = head(z)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
        scheduler.step()

        # Eval (채널 없이 순수 분류 성능 측정 — 학습 안정성 확인용)
        encoder.eval(); head.eval()
        correct, total, total_bytes = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                z = encoder(x)
                z_cpu = z.cpu()
                z_q, z_min, z_max = quantizer.quantize(z_cpu)
                total_bytes += quantizer.byte_count(z_cpu)
                z_deq = quantizer.dequantize(z_q, z_min, z_max).to(device)
                logits = head(z_deq)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_bytes = total_bytes / total
        logger.info(f"Epoch {epoch:3d} | loss {total_loss/len(train_loader.dataset):.4f} | acc {acc:.4f} | avg_bytes {avg_bytes:.1f}")
        results.append({"epoch": epoch, "test_acc": acc, "avg_bytes": avg_bytes})

        if acc > best_acc:
            best_acc = acc
            ckpt_path = f"experiments/checkpoints/semantic_enc_dim{cfg.model.latent_dim}.pth"
            save_checkpoint(
                {"epoch": epoch, "encoder_state_dict": encoder.state_dict(),
                 "head_state_dict": head.state_dict(), "acc": acc,
                 "latent_dim": cfg.model.latent_dim},
                ckpt_path,
            )

    logger.info(f"Best accuracy: {best_acc:.4f}")

    os.makedirs("experiments/results", exist_ok=True)
    out_path = f"experiments/results/semantic_dim{cfg.model.latent_dim}_training.json"
    with open(out_path, "w") as f:
        json.dump({"best_acc": best_acc, "latent_dim": cfg.model.latent_dim, "history": results}, f, indent=2)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_semantic.yaml")
    parser.add_argument("--latent_dim", type=int, default=None)
    args = parser.parse_args()

    base = OmegaConf.load("configs/default.yaml")
    override = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base, override)
    if args.latent_dim is not None:
        cfg.model.latent_dim = args.latent_dim

    train(cfg)
