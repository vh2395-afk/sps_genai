"""Train the Energy-Based Model on CIFAR-10 with contrastive divergence.

Run from the repo root:
    uv run python train_ebm.py --epochs 10

Each batch:
  1. Draw negative samples: 95% from a replay buffer, 5% fresh uniform noise.
  2. Run 60 Langevin steps on them (gradients w.r.t. the images, weights frozen).
  3. Update the weights so real images get LOW energy and samples get HIGH energy:
        loss = E(real) - E(fake) + alpha * (E(real)^2 + E(fake)^2)

Outputs:
    models/ebm.pth              weights loaded by app/image_gen_service.py
    samples/ebm_epoch_XX.png    8x8 grid of Langevin samples after each epoch
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.ebm import IMG_SIZE, EnergyModel, langevin_sample

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "ebm.pth"
SAMPLE_DIR = ROOT / "samples"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReplayBuffer:
    def __init__(self, size: int = 8192):
        self.size = size
        self.examples = [self._noise(1) for _ in range(128)]

    @staticmethod
    def _noise(n: int) -> torch.Tensor:
        return torch.rand(n, 3, IMG_SIZE, IMG_SIZE) * 2 - 1

    def draw(self, n: int) -> torch.Tensor:
        n_new = max(1, int(n * 0.05))
        old = torch.cat(random.choices(self.examples, k=n - n_new), dim=0)
        return torch.cat([self._noise(n_new), old], dim=0)

    def add(self, x: torch.Tensor) -> None:
        self.examples = list(x.cpu().chunk(x.size(0), dim=0)) + self.examples
        self.examples = self.examples[: self.size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--langevin-steps", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])
    dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    model = EnergyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.0, 0.999))
    buffer = ReplayBuffer()
    SAMPLE_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = sum_real = sum_fake = 0.0

        for real, _ in loader:
            real = (real + 0.005 * torch.randn_like(real)).clamp(-1, 1).to(device)

            fake = buffer.draw(real.size(0)).to(device)
            fake = langevin_sample(model, fake, steps=args.langevin_steps)
            buffer.add(fake)

            e_real = model(real)
            e_fake = model(fake)
            cd_loss = e_real.mean() - e_fake.mean()
            reg_loss = args.alpha * (e_real.pow(2).mean() + e_fake.pow(2).mean())
            loss = cd_loss + reg_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()

            sum_loss += loss.item()
            sum_real += e_real.mean().item()
            sum_fake += e_fake.mean().item()

        n = len(loader)
        print(f"Epoch {epoch:2d} | loss {sum_loss / n:.4f} | "
              f"E(real) {sum_real / n:.3f} | E(fake) {sum_fake / n:.3f}")

        g = torch.Generator().manual_seed(0)
        start = (torch.rand(64, 3, IMG_SIZE, IMG_SIZE, generator=g) * 2 - 1).to(device)
        samples = langevin_sample(model, start, steps=256)
        save_image(samples, SAMPLE_DIR / f"ebm_epoch_{epoch:02d}.png",
                   nrow=8, normalize=True, value_range=(-1, 1))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved energy model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
