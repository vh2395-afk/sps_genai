"""FastAPI routes for the MNIST GAN.

Wire into app/main.py with:
    from app.gan_router import router as gan_router
    app.include_router(gan_router)
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.gan_service import generate_png

router = APIRouter(prefix="/gan", tags=["MNIST GAN"])


@router.get(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Grid of generated digits"}},
)
def generate_digits(
    num_images: int = Query(16, ge=1, le=64, description="How many digits to generate"),
    seed: int | None = Query(None, description="Optional seed for reproducible output"),
):
    try:
        png = generate_png(num_images=num_images, seed=seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return Response(content=png, media_type="image/png")
