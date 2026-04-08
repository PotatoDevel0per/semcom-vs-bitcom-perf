"""ResNet-18 분류기 사전학습 스크립트."""
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
from src.models.classifier import build_classifier


def train(cfg):
    set_seed(cfg.seed)
    logger = get_logger("train_classifier")
    device = torch.device(cfg.device)

    train_loader, test_loader = get_loaders(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = build_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=0.9,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    best_acc = 0.0
    results = []

    for epoch in range(1, cfg.train.epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
        scheduler.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total

        logger.info(f"Epoch {epoch:3d} | loss {total_loss/len(train_loader.dataset):.4f} | test acc {acc:.4f}")
        results.append({"epoch": epoch, "test_acc": acc})

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "acc": acc},
                cfg.output.checkpoint,
            )

    logger.info(f"Best test accuracy: {best_acc:.4f}")

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/classifier_training.json", "w") as f:
        json.dump({"best_acc": best_acc, "history": results}, f, indent=2)

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier.yaml")
    args = parser.parse_args()

    base = OmegaConf.load("configs/default.yaml")
    override = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base, override)

    train(cfg)
