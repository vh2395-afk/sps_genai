"""Train the Assignment 3 GAN on MNIST and save the generator for the API.

Run from the repo root (same folder as train_cnn.py):
    uv run python train_gan.py --epochs 20

Outputs:
    models/gan_generator.pth        generator weights loaded by app/gan_service.py
    samples/gan_epoch_XX.png        8x8 grid of generated digits after each epoch
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan import LATENT_DIM, Discriminator, Generator, weights_init

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "gan_generator.pth"
SAMPLE_DIR = ROOT / "samples"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Scale pixels to [-1, 1] to match the generator's Tanh output range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)  # same z every epoch
    SAMPLE_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        sum_d = sum_g = 0.0

        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # --- Discriminator: maximize log D(x) + log(1 - D(G(z))) ---
            z = torch.randn(b, LATENT_DIM, device=device)
            fake = G(z)
            loss_D = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator: non-saturating loss, maximize log D(G(z)) ---
            loss_G = criterion(D(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            sum_d += loss_D.item()
            sum_g += loss_G.item()

        n = len(loader)
        print(f"Epoch {epoch:2d} | D loss {sum_d / n:.4f} | G loss {sum_g / n:.4f}")

        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, SAMPLE_DIR / f"gan_epoch_{epoch:02d}.png",
                   nrow=8, normalize=True, value_range=(-1, 1))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), MODEL_PATH)
    print(f"Saved generator to {MODEL_PATH}")
    print(f"Sample grids in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
