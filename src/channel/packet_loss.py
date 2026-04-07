import torch
from .base import BaseChannel


class PacketLossChannel(BaseChannel):
    """블록 단위 패킷 손실 시뮬레이터."""

    def __init__(self, loss_rate: float, block_size: int = 1):
        self.loss_rate = loss_rate
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.loss_rate == 0.0:
            return x
        # x: (batch, latent_dim) 또는 (latent_dim,)
        flat = x.reshape(x.shape[0], -1) if x.dim() > 1 else x.unsqueeze(0)
        num_blocks = flat.shape[1] // self.block_size
        mask = (torch.rand(flat.shape[0], num_blocks, device=x.device) > self.loss_rate).float()
        mask = mask.repeat_interleave(self.block_size, dim=1)
        mask = mask[:, :flat.shape[1]]
        return (flat * mask).reshape(x.shape)
