import torch
import numpy as np
from typing import Callable


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def accuracy_per_bit(acc: float, byte_count: float) -> float:
    bits = byte_count * 8
    return acc / bits if bits > 0 else 0.0


def evaluate_epoch(model: torch.nn.Module, dataloader, device: str,
                   preprocess_fn: Callable = None) -> dict:
    """
    preprocess_fn: 이미지 텐서를 받아 분류기 입력 텐서를 반환하는 함수.
                   None이면 원시 이미지를 그대로 사용.
    """
    model.eval()
    total_correct, total, total_bytes = 0, 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if preprocess_fn is not None:
                x, byte_cnt = preprocess_fn(x)
                total_bytes += byte_cnt
            logits = model(x)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    acc = total_correct / total
    avg_bytes = total_bytes / total if total_bytes > 0 else None
    result = {"top1_acc": acc}
    if avg_bytes is not None:
        result["avg_bytes"] = avg_bytes
        result["accuracy_per_bit"] = accuracy_per_bit(acc, avg_bytes)
    return result


def repeated_eval(eval_fn: Callable, repeat: int = 5, base_seed: int = 42) -> dict:
    """eval_fn을 repeat회 실행하여 평균/표준편차를 반환."""
    results = [eval_fn(seed=base_seed + i) for i in range(repeat)]
    keys = results[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in results])
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std())
    return summary
