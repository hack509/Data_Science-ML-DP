import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)

SMALL = ["le colis est arrivé", "le colis est en attente, payez les frais", "payez vite, offre gratuite"]


def test_tokenize_and_vocabulary():
    assert ex.tokenize("L'écran EST bien, 2 fois !") == ["écran", "est", "bien", "fois"]
    vocab = ex.fit_vocabulary([["b", "a"], ["c", "a"]])
    assert vocab == {"a": 0, "b": 1, "c": 2}


def test_counts_ignore_unknown_words():
    counts = ex.term_counts([["a", "a", "z"]], {"a": 0, "b": 1})
    assert counts.tolist() == [[2.0, 0.0]]


def test_matches_sklearn_on_small_corpus():
    reference = TfidfVectorizer().fit(SMALL)
    ours = ex.TfidfFromScratch().fit(SMALL)
    assert ours.vocabulary_ == reference.vocabulary_
    np.testing.assert_allclose(ours.idf_, reference.idf_)
    np.testing.assert_allclose(ours.transform(SMALL), reference.transform(SMALL).toarray(), atol=1e-10)


def test_matches_sklearn_on_reviews():
    texts = load_csv("reviews")["text"].tolist()
    ours = ex.TfidfFromScratch().fit_transform(texts)
    reference = TfidfVectorizer().fit_transform(texts).toarray()
    np.testing.assert_allclose(ours, reference, atol=1e-10)


def test_unknown_document_is_zero_and_search_works():
    model = ex.TfidfFromScratch().fit(SMALL)
    assert not model.transform(["zzz qqq"]).any()
    matrix = model.transform(SMALL)
    results = model.most_similar("colis en attente", matrix, k=2)
    assert results[0][0] == 1 and results[0][1] >= results[1][1]
    reviews = load_csv("reviews")["text"].tolist()
    big = ex.TfidfFromScratch().fit(reviews)
    best_index, score = big.most_similar("livraison rapide", big.transform(reviews), k=1)[0]
    assert "livraison rapide" in reviews[best_index].lower() and score > 0
