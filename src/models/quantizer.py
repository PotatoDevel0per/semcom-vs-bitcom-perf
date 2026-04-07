import numpy as np
import torch


class ScalarQuantizer:
    """8-bit 균일 양자화. 잠재 벡터를 uint8로 변환하여 바이트 수를 계산한다."""

    def quantize(self, z: torch.Tensor):
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)
        z_q = (z_norm * 255).round().clamp(0, 255).to(torch.uint8)
        return z_q, z_min, z_max

    def dequantize(self, z_q: torch.Tensor, z_min, z_max) -> torch.Tensor:
        return z_q.float() / 255.0 * (z_max - z_min) + z_min

    def byte_count(self, z: torch.Tensor) -> int:
        return z.numel()  # uint8 1byte per element


class VectorQuantizer:
    """K-means 기반 벡터 양자화."""

    def __init__(self, num_codes: int = 256):
        self.num_codes = num_codes
        self.codebook = None

    def fit(self, z: np.ndarray) -> None:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=self.num_codes, random_state=42)
        km.fit(z)
        self.codebook = km.cluster_centers_
        self._km = km

    def quantize(self, z: np.ndarray) -> np.ndarray:
        return self._km.predict(z)  # 인덱스 배열

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        return self.codebook[indices]

    def bits_per_vector(self) -> int:
        return int(np.ceil(np.log2(self.num_codes)))  # bits per vector
