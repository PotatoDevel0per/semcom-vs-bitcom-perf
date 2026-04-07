import torch


class RawTransmission:
    """원시 이미지 픽셀을 채널을 통해 전송하는 baseline."""

    CIFAR10_BYTES = 32 * 32 * 3  # 3072 bytes

    def __init__(self, channel=None):
        self.channel = channel

    def transmit(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel is not None:
            return self.channel(x)
        return x

    def byte_count(self) -> int:
        return self.CIFAR10_BYTES
