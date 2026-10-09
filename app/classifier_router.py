"""FastAPI routes for the CIFAR-10 classifier.

Wire into app/main.py with:
    from app.classifier_router import router as classifier_router
    app.include_router(classifier_router)
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.classifier import classify_image
from app.cnn import CIFAR10_CLASSES

router = APIRouter(prefix="/classifier", tags=["CIFAR-10 classifier"])


@router.get("/classes")
def get_classes():
    return {"classes": CIFAR10_CLASSES}


@router.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (png/jpg).")
    if not 1 <= top_k <= len(CIFAR10_CLASSES):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10.")

    try:
        return classify_image(await file.read(), top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")
