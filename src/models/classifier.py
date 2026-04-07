import torch.nn as nn
from torchvision.models import resnet18


def build_classifier(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    model = resnet18(pretrained=pretrained)
    # CIFAR-10: 32x32 입력에 맞게 첫 conv와 maxpool 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
