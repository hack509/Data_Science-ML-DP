"""Reference solution - A03 TF-IDF from scratch, identical to scikit-learn's default TfidfVectorizer."""

from __future__ import annotations

import re

import numpy as np

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn default: words of 2+ characters


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def fit_vocabulary(tokenized_documents: list[list[str]]) -> dict[str, int]:
    """Alphabetically sorted vocabulary, as in scikit-learn."""
    words = sorted({token for document in tokenized_documents for token in document})
    return {word: index for index, word in enumerate(words)}


def term_counts(tokenized_documents: list[list[str]], vocabulary: dict[str, int]) -> np.ndarray:
    counts = np.zeros((len(tokenized_documents), len(vocabulary)))
    for row, document in enumerate(tokenized_documents):
        for token in document:
            column = vocabulary.get(token)
            if column is not None:  # unknown words are ignored
                counts[row, column] += 1
    return counts


def inverse_document_frequency(counts: np.ndarray) -> np.ndarray:
    """Smoothed idf: ln((1 + n) / (1 + df)) + 1."""
    n_documents = counts.shape[0]
    document_frequency = (counts > 0).sum(axis=0)
    return np.log((1 + n_documents) / (1 + document_frequency)) + 1


def tfidf(counts: np.ndarray, idf: np.ndarray) -> np.ndarray:
    weights = counts * idf
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # documents without known words stay at zero
    return weights / norms


class TfidfFromScratch:
    def fit(self, documents: list[str]) -> "TfidfFromScratch":
        tokenized = [tokenize(d) for d in documents]
        self.vocabulary_ = fit_vocabulary(tokenized)
        self.idf_ = inverse_document_frequency(term_counts(tokenized, self.vocabulary_))
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        return tfidf(term_counts([tokenize(d) for d in documents], self.vocabulary_), self.idf_)

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def most_similar(self, query: str, documents_matrix: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
        """Indices and cosine similarities of the k documents closest to the query."""
        scores = documents_matrix @ self.transform([query])[0]
        best = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in best]
