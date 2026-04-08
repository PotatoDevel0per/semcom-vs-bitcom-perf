"""Raw / JPEG baseline 평가 스크립트."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from omegaconf import OmegaConf

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.checkpoint import load_checkpoint
from src.data.cifar10 import get_loaders
from src.models.classifier import build_classifier
from src.baseline.jpeg import jpeg_compress_batch


def eval_raw(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    byte_count = 32 * 32 * 3
    acc = correct / total
    return {"method": "raw", "byte_budget": byte_count,
            "top1_acc": acc, "accuracy_per_bit": acc / (byte_count * 8)}


def eval_jpeg(model, loader, device, quality):
    model.eval()
    correct, total, total_bytes = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x_rec, sizes = jpeg_compress_batch(x, quality=quality)
            x_rec, y = x_rec.to(device), y.to(device)
            correct += (model(x_rec).argmax(1) == y).sum().item()
            total += y.size(0)
            total_bytes += sum(sizes)
    acc = correct / total
    avg_bytes = total_bytes / total
    return {"method": f"jpeg_q{quality}", "byte_budget": avg_bytes,
            "top1_acc": acc, "accuracy_per_bit": acc / (avg_bytes * 8)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    logger = get_logger("eval_baseline")
    device = torch.device(cfg.device)

    _, test_loader = get_loaders(cfg.data.root, batch_size=cfg.data.batch_size,
                                  num_workers=cfg.data.num_workers)

    model = build_classifier().to(device)
    load_checkpoint("experiments/checkpoints/classifier_resnet18.pth", model)
    logger.info("Classifier loaded.")

    results = []

    # Raw baseline
    r = eval_raw(model, test_loader, device)
    logger.info(f"Raw: acc={r['top1_acc']:.4f}, bytes={r['byte_budget']}")
    results.append(r)

    # JPEG at various quality levels → 비트 예산별
    # quality 값을 조정해 비트 예산에 대응
    for q in [1, 3, 5, 10, 20, 40, 60, 80, 95]:
        r = eval_jpeg(model, test_loader, device, quality=q)
        logger.info(f"JPEG q={q}: acc={r['top1_acc']:.4f}, avg_bytes={r['byte_budget']:.1f}, acc/bit={r['accuracy_per_bit']:.6f}")
        results.append(r)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to experiments/results/baseline_eval.json")
