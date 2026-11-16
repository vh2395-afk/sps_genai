"""Inference for the diffusion model and the energy-based model (CPU)."""

import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from app.diffusion import Diffusion, UNet
from app.ebm import IMG_SIZE, EnergyModel, langevin_sample

MODELS = Path(__file__).resolve().parent.parent / "models"
DIFFUSION_PATH = MODELS / "diffusion_unet.pth"
EBM_PATH = MODELS / "ebm.pth"


def _load(model: torch.nn.Module, path: Path, train_cmd: str) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"No trained weights at {path}. Run `uv run python {train_cmd}` first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


@lru_cache(maxsize=1)
def load_unet() -> UNet:
    return _load(UNet(), DIFFUSION_PATH, "train_diffusion.py")


@lru_cache(maxsize=1)
def load_ebm() -> EnergyModel:
    return _load(EnergyModel(), EBM_PATH, "train_ebm.py")


def to_png(imgs: torch.Tensor, scale: int = 4) -> bytes:
    """imgs in [-1, 1], shape (N, 3, H, W) -> PNG grid bytes."""
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    grid = make_grid(imgs, nrow=min(8, imgs.size(0)), padding=2, pad_value=1.0)
    arr = (grid.permute(1, 2, 0) * 255).byte().numpy()
    img = Image.fromarray(arr)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gen(seed: int | None) -> torch.Generator | None:
    return torch.Generator().manual_seed(seed) if seed is not None else None


def diffusion_png(num_images: int, steps: int, seed: int | None) -> bytes:
    imgs = Diffusion(device="cpu").sample(load_unet(), num_images, steps=steps, generator=_gen(seed))
    return to_png(imgs)


def ebm_png(num_images: int, steps: int, seed: int | None) -> bytes:
    start = torch.rand(num_images, 3, IMG_SIZE, IMG_SIZE, generator=_gen(seed)) * 2 - 1
    imgs = langevin_sample(load_ebm(), start, steps=steps)
    return to_png(imgs)
