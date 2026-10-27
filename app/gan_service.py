"""Inference wrapper: load the trained generator and produce digit images."""

import io
from functools import lru_cache
from pathlib import Path

import torch
from torchvision.utils import make_grid
from PIL import Image

from app.gan import LATENT_DIM, Generator

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gan_generator.pth"


@lru_cache(maxsize=1)
def load_generator() -> Generator:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained generator at {MODEL_PATH}. Run `uv run python train_gan.py` first."
        )
    G = Generator()
    G.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    G.eval()  # BatchNorm uses running stats, so a single image works
    return G


def generate_png(num_images: int = 16, seed: int | None = None) -> bytes:
    """Generate `num_images` digits and return them as one PNG grid."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(num_images, LATENT_DIM, generator=gen)

    with torch.no_grad():
        imgs = load_generator()(z)                    # (N, 1, 28, 28) in [-1, 1]

    imgs = (imgs + 1) / 2                             # -> [0, 1]
    nrow = min(8, num_images)
    grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=1.0)  # (3, H, W)
    arr = (grid.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()

    img = Image.fromarray(arr)
    img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)  # easier to see
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
