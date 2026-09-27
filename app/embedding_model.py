"""Word-embedding lookups backed by spaCy's static word vectors.

Uses en_core_web_md (300-dim GloVe-style vectors, ~20k unique vectors). The small
model (en_core_web_sm) does NOT ship real word vectors, so md or lg is required.
"""

from __future__ import annotations

import numpy as np
import spacy


class EmbeddingModel:
    def __init__(self, model_name: str = "en_core_web_md"):
        # Only the vocab/vectors are needed, so skip the heavier pipeline components.
        self.nlp = spacy.load(model_name, exclude=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
        self.model_name = model_name
        self.dim = self.nlp.vocab.vectors_length

    def _lookup(self, word: str):
        """Return the Lexeme for a word, falling back to lowercase if the exact form has no vector."""
        word = word.strip()
        lex = self.nlp.vocab[word]
        if not lex.has_vector:
            lex = self.nlp.vocab[word.lower()]
        return lex

    def has_vector(self, word: str) -> bool:
        return self._lookup(word).has_vector

    def get_embedding(self, word: str, normalize: bool = False) -> np.ndarray | None:
        lex = self._lookup(word)
        if not lex.has_vector:
            return None
        vec = lex.vector.astype(np.float32)
        if normalize:
            norm = np.linalg.norm(vec)
            vec = vec / norm if norm > 0 else vec
        return vec

    def similarity(self, word1: str, word2: str) -> float | None:
        v1, v2 = self.get_embedding(word1), self.get_embedding(word2)
        if v1 is None or v2 is None:
            return None
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0

    def most_similar(self, word: str, top_n: int = 10) -> list[dict] | None:
        vec = self.get_embedding(word)
        if vec is None:
            return None
        # Ask for extra results so we can drop the query word and its case variants.
        keys, _, scores = self.nlp.vocab.vectors.most_similar(np.asarray([vec]), n=top_n + 10)
        query = word.strip().lower()
        results, seen = [], set()
        for key, score in zip(keys[0], scores[0]):
            text = self.nlp.vocab.strings[int(key)]
            norm_text = text.lower()
            if norm_text == query or norm_text in seen:
                continue
            seen.add(norm_text)
            results.append({"word": text, "similarity": round(float(score), 4)})
            if len(results) == top_n:
                break
        return results
