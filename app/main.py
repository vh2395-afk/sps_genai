from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.classifier_router import router as classifier_router
from app.gan_router import router as gan_router
from app.image_gen_router import diffusion_router, ebm_router

app = FastAPI()
app.include_router(classifier_router)
app.include_router(gan_router)
app.include_router(diffusion_router)
app.include_router(ebm_router)

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel("en_core_web_md")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vec = embedding_model.get_embedding(request.word)
    if vec is None:
        raise HTTPException(status_code=404, detail=f"No embedding for '{request.word}'")
    return {"word": request.word, "dimension": len(vec), "embedding": vec.tolist()}