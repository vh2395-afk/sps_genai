"""Train the diffusion UNet on CIFAR-10 and save it for the API.

Run from the repo root:
    uv run python train_diffusion.py --epochs 20

Outputs:
    models/diffusion_unet.pth        EMA weights loaded by app/image_gen_service.py
    samples/diffusion_epoch_XX.png   8x8 grid of samples (every --sample-every epochs and at the end)
"""

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.diffusion import Diffusion, UNet

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "diffusion_unet.pth"
SAMPLE_DIR = ROOT / "samples"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample-every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])
    dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    model = UNet().to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    diffusion = Diffusion(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    SAMPLE_DIR.mkdir(exist_ok=True)
    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            loss = diffusion.loss(model, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(ema, model)
            total += loss.item()

        print(f"Epoch {epoch:2d} | loss {total / len(loader):.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            g = torch.Generator().manual_seed(0)
            imgs = diffusion.sample(ema, 64, steps=50, generator=g)
            save_image(imgs, SAMPLE_DIR / f"diffusion_epoch_{epoch:02d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema.state_dict(), MODEL_PATH)
    print(f"Saved diffusion model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
