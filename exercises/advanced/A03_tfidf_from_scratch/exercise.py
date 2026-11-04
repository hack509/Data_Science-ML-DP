"""A03 - TF-IDF from scratch (NumPy only). Complete the code (see README.md)."""

from __future__ import annotations

import re

import numpy as np

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    raise NotImplementedError("TODO")


def fit_vocabulary(tokenized_documents: list[list[str]]) -> dict[str, int]:
    raise NotImplementedError("TODO")


def term_counts(tokenized_documents: list[list[str]], vocabulary: dict[str, int]) -> np.ndarray:
    raise NotImplementedError("TODO")


def inverse_document_frequency(counts: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def tfidf(counts: np.ndarray, idf: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


class TfidfFromScratch:
    def fit(self, documents: list[str]) -> "TfidfFromScratch":
        raise NotImplementedError("TODO")

    def transform(self, documents: list[str]) -> np.ndarray:
        raise NotImplementedError("TODO")

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        raise NotImplementedError("TODO")

    def most_similar(self, query: str, documents_matrix: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
        raise NotImplementedError("TODO")
