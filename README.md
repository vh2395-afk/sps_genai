# sps_genai – FastAPI + spaCy Word Embeddings

FastAPI service with a bigram text generator (Module 3) and a spaCy word-embedding endpoint (Assignment 1).

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | – | health check |
| POST | `/generate` | `{"start_word": "the", "length": 10}` | bigram-generated text |
| POST | `/embedding` | `{"word": "king"}` | 300-dim embedding from `en_core_web_md` |

Unknown words return **404**.

## Project structure

```
app/
├── main.py              # FastAPI routes
├── bigram_model.py      # bigram text generator
└── embedding_model.py   # spaCy embedding lookup
Dockerfile
pyproject.toml / uv.lock
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Open http://127.0.0.1:8000/docs

## Example

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "king"}'
```
