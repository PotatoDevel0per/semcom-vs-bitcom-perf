import torch
import torch.nn as nn
from torchvision.models import resnet18


class SemanticEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        base = resnet18(pretrained=False)
        # CIFAR-10용 수정
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        # avgpool 이전까지를 feature extractor로 사용
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.proj = nn.Linear(512, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        feat = feat.flatten(1)
        return self.proj(feat)
