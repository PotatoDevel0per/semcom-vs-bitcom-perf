import torch
from .base import BaseChannel


class RayleighChannel(BaseChannel):
    """Rayleigh 페이딩 채널 + AWGN. Perfect CSI 가정 하에 등화."""

    def __init__(self, snr_db: float):
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rayleigh 계수: h ~ CN(0,1) 실수부만 사용
        h = torch.randn(1, device=x.device)
        y = h * x

        snr_linear = 10 ** (self.snr_db / 10)
        signal_power = (h * x).pow(2).mean()
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(y) * noise_power.sqrt()
        y = y + noise

        # Perfect CSI equalization
        return y / (h + 1e-8)
