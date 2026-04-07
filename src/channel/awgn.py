import torch
from .base import BaseChannel


class AWGNChannel(BaseChannel):
    def __init__(self, snr_db: float):
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.snr_db == float("inf"):
            return x
        snr_linear = 10 ** (self.snr_db / 10)
        signal_power = x.pow(2).mean()
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise
