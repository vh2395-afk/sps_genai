"""GAN architecture for Assignment 3 (Part 1).

Generator (noise -> image):
    Input z: (B, 100)
    Linear(100 -> 7*7*128), reshape to (B, 128, 7, 7)
    ConvTranspose2d(128 -> 64, k=4, s=2, p=1) -> BatchNorm2d -> ReLU   # (B, 64, 14, 14)
    ConvTranspose2d(64 -> 1,   k=4, s=2, p=1) -> Tanh                   # (B, 1, 28, 28)

Discriminator (image -> probability real):
    Input x: (B, 1, 28, 28)
    Conv2d(1 -> 64,   k=4, s=2, p=1) -> LeakyReLU(0.2)                  # (B, 64, 14, 14)
    Conv2d(64 -> 128, k=4, s=2, p=1) -> BatchNorm2d -> LeakyReLU(0.2)   # (B, 128, 7, 7)
    Flatten -> Linear(128*7*7 -> 1) -> Sigmoid                          # (B, 1)

Shape check:  uv run python -m app.gan
"""

import torch
import torch.nn as nn

LATENT_DIM = 100
IMG_SIZE = 28


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),     # 28 -> 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # output is P(real)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weights_init(m: nn.Module) -> None:
    """DCGAN-style init: N(0, 0.02) for conv weights, N(1, 0.02) for BatchNorm."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    G, D = Generator(), Discriminator()
    z = torch.randn(4, LATENT_DIM)
    fake = G(z)
    out = D(fake)
    print(G)
    print(D)
    print("generator output:", tuple(fake.shape))      # (4, 1, 28, 28)
    print("discriminator output:", tuple(out.shape))   # (4, 1)
    print("G params:", sum(p.numel() for p in G.parameters()))
    print("D params:", sum(p.numel() for p in D.parameters()))
