import torch


class BaseChannel:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
