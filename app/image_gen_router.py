"""FastAPI routes for the CIFAR-10 diffusion model and energy-based model.

Wire into app/main.py with:
    from app.image_gen_router import diffusion_router, ebm_router
    app.include_router(diffusion_router)
    app.include_router(ebm_router)
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.image_gen_service import diffusion_png, ebm_png

PNG = {200: {"content": {"image/png": {}}, "description": "Grid of generated 32x32 images"}}

diffusion_router = APIRouter(prefix="/diffusion", tags=["CIFAR-10 Diffusion"])
ebm_router = APIRouter(prefix="/ebm", tags=["CIFAR-10 Energy-Based Model"])


@diffusion_router.get("/generate", response_class=Response, responses=PNG)
def diffusion_generate(
    num_images: int = Query(16, ge=1, le=64),
    steps: int = Query(50, ge=10, le=1000, description="DDIM denoising steps"),
    seed: int | None = Query(None, description="Optional seed for reproducible output"),
):
    try:
        return Response(content=diffusion_png(num_images, steps, seed), media_type="image/png")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@ebm_router.get("/generate", response_class=Response, responses=PNG)
def ebm_generate(
    num_images: int = Query(16, ge=1, le=64),
    steps: int = Query(256, ge=10, le=1000, description="Langevin dynamics steps"),
    seed: int | None = Query(None, description="Optional seed for reproducible output"),
):
    try:
        return Response(content=ebm_png(num_images, steps, seed), media_type="image/png")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
