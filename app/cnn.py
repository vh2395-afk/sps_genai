"""CNN architecture for Assignment 2 (Part 1) and the shared preprocessing used
for CIFAR-10 training and API inference (Part 2).

Architecture (input 64x64x3):
    Conv2d(3->16, 3x3, s=1, p=1) -> ReLU -> MaxPool(2, 2)     # 16 x 32 x 32
    Conv2d(16->32, 3x3, s=1, p=1) -> ReLU -> MaxPool(2, 2)    # 32 x 16 x 16
    Flatten                                                   # 8192
    Linear(8192 -> 100) -> ReLU
    Linear(100 -> 10)
"""

import torch
import torch.nn as nn
from torchvision import transforms

IMG_SIZE = 64
NUM_CLASSES = 10

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 per-channel statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# CIFAR-10 images are 32x32, so they are upsampled to the 64x64 input the
# assignment's architecture expects. The same transform is used at inference.
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


if __name__ == "__main__":
    # Quick shape check: python -m app.cnn
    model = SimpleCNN()
    out = model(torch.randn(4, 3, IMG_SIZE, IMG_SIZE))
    print(model)
    print("output shape:", tuple(out.shape))  # (4, 10)
    print("params:", sum(p.numel() for p in model.parameters()))
