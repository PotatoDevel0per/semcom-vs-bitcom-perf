import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.data.transforms import CIFAR10_MEAN, CIFAR10_STD

_denorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
    std=[1.0 / s for s in CIFAR10_STD],
)
_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()
_renorm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def _compress_decompress(img_tensor: torch.Tensor, quality: int):
    """단일 이미지 텐서를 JPEG 압축 후 복원."""
    img = _to_pil(_denorm(img_tensor).clamp(0, 1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    byte_size = buf.tell()
    buf.seek(0)
    img_rec = Image.open(buf).convert("RGB")
    return _renorm(_to_tensor(img_rec)), byte_size


def jpeg_compress_batch(x: torch.Tensor, quality: int):
    """배치 전체를 JPEG 압축/복원. (batch, C, H, W) 반환."""
    results, sizes = [], []
    for img in x:
        rec, sz = _compress_decompress(img.cpu(), quality)
        results.append(rec)
        sizes.append(sz)
    return torch.stack(results).to(x.device), sizes


def find_quality_for_budget(sample_img: torch.Tensor, target_bytes: int) -> int:
    """target_bytes에 가장 근접한 JPEG quality 값을 반환."""
    best_q, best_diff = 1, float("inf")
    for q in range(1, 96):
        _, sz = _compress_decompress(sample_img, q)
        diff = abs(sz - target_bytes)
        if diff < best_diff:
            best_diff, best_q = diff, q
    return best_q
