"""CIFAR-10 DataLoader 팩토리."""
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from src.data.transforms import train_transform, test_transform


def get_loaders(root: str, batch_size: int = 128,
                num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    train_set = CIFAR10(root=root, train=True, download=True,
                        transform=train_transform())
    test_set = CIFAR10(root=root, train=False, download=True,
                       transform=test_transform())

    pin_memory = num_workers > 0
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, test_loader
