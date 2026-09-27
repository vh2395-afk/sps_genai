import random
import re
from collections import Counter, defaultdict


class BigramModel:
    """Simple bigram language model: P(next word | current word)."""

    def __init__(self, corpus: list[str]):
        self.bigram_counts: dict[str, Counter] = defaultdict(Counter)
        self._build(corpus)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _build(self, corpus: list[str]) -> None:
        for document in corpus:
            tokens = self.tokenize(document)
            for current, following in zip(tokens, tokens[1:]):
                self.bigram_counts[current][following] += 1

    def next_word(self, word: str) -> str | None:
        followers = self.bigram_counts.get(word.lower())
        if not followers:
            return None
        words, counts = zip(*followers.items())
        return random.choices(words, weights=counts, k=1)[0]

    def generate_text(self, start_word: str, length: int = 10) -> str:
        current = start_word.lower()
        generated = [current]
        for _ in range(max(0, length - 1)):
            nxt = self.next_word(current)
            if nxt is None:
                break
            generated.append(nxt)
            current = nxt
        return " ".join(generated)