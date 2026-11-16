"""Energy-Based Model for Assignment 4 (CIFAR-10, 32x32x3).

EnergyModel maps an image to a scalar energy E(x): low energy = realistic image.
Sampling uses Langevin dynamics: we freeze the network weights (requires_grad=False)
and instead make the *input image* require gradients, then step the image downhill
on the energy surface:
    x <- x - step_size * dE/dx + noise

Shape check:  uv run python -m app.ebm
"""

import torch
import torch.nn as nn

IMG_SIZE = 32


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        c1, c2, c3 = hidden, hidden * 2, hidden * 4
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=1, padding=1), Swish(),    # 32x32
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), Swish(),   # 16x16
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), Swish(),   # 8x8
            nn.Conv2d(c3, c3, 4, stride=2, padding=1), Swish(),   # 4x4
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, c3), Swish(),
            nn.Linear(c3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # energy per image, shape (B,). Use [:, 0] rather than .squeeze(-1): on Apple
        # GPUs (MPS), squeeze's backward returns zero input-gradients for most of the
        # batch, which silently breaks Langevin sampling.
        return self.net(x)[:, 0]


def langevin_sample(model: nn.Module, x: torch.Tensor, steps: int = 60,
                    step_size: float = 10.0, noise_std: float = 0.005) -> torch.Tensor:
    """Run Langevin dynamics on the input images to lower their energy.

    Gradients are taken w.r.t. the INPUT x, not the model parameters.
    """
    was_training = model.training
    model.eval()
    for p in model.parameters():           # weights do not contribute to this descent
        p.requires_grad_(False)

    x = x.detach().clone()
    with torch.enable_grad():               # works even if the caller is inside no_grad
        for _ in range(steps):
            x = (x + noise_std * torch.randn_like(x)).clamp(-1, 1)
            x.requires_grad_(True)          # the image is what we optimize
            energy = model(x).sum()
            (grad,) = torch.autograd.grad(energy, x)
            x = (x - step_size * grad.clamp(-0.03, 0.03)).detach().clamp(-1, 1)

    for p in model.parameters():
        p.requires_grad_(True)
    model.train(was_training)
    return x


if __name__ == "__main__":
    model = EnergyModel()
    x = torch.rand(4, 3, IMG_SIZE, IMG_SIZE) * 2 - 1
    print(model)
    print("energy output:", tuple(model(x).shape))     # (4,)
    print("params:", sum(p.numel() for p in model.parameters()))
    print("after 5 Langevin steps:", tuple(langevin_sample(model, x, steps=5).shape))
