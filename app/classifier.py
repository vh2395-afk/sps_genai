"""Inference wrapper around the trained CIFAR-10 SimpleCNN."""

import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from app.cnn import CIFAR10_CLASSES, SimpleCNN, eval_transform

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.pth"


@lru_cache(maxsize=1)
def load_model() -> SimpleCNN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained weights at {MODEL_PATH}. Run `uv run python train_cnn.py` first."
        )
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def classify_image(image_bytes: bytes, top_k: int = 3) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = eval_transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(load_model()(x), dim=1).squeeze(0)

    top_probs, top_idx = probs.topk(top_k)
    return {
        "prediction": CIFAR10_CLASSES[top_idx[0].item()],
        "confidence": round(top_probs[0].item(), 4),
        "top_k": [
            {"label": CIFAR10_CLASSES[i.item()], "probability": round(p.item(), 4)}
            for p, i in zip(top_probs, top_idx)
        ],
    }
