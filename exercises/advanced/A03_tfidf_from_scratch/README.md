# A03 — TF-IDF implémenté de zéro

- **Identifiant** : A03
- **Niveau** : avancé
- **Module** : 8 (leçon 8.2)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Comprendre exactement ce que calcule `TfidfVectorizer` (tokenisation, vocabulaire, IDF lissé, normalisation L2).
- Reproduire une bibliothèque au chiffre près et le vérifier par des tests.
- Utiliser TF-IDF pour une recherche de documents par similarité cosinus.

## Énoncé

Complétez `exercise.py` avec NumPy (sans scikit-learn) pour obtenir **exactement** les résultats de `TfidfVectorizer()` avec ses paramètres par défaut :

1. `tokenize(text)` : minuscules, puis expression régulière `(?u)\b\w\w+\b` (mots d'au moins 2 caractères).
2. `fit_vocabulary(tokenized_documents)` : `{mot: indice}` avec les mots triés par ordre alphabétique.
3. `term_counts(tokenized_documents, vocabulary)` : matrice (documents × mots) des comptages ; les mots inconnus sont ignorés.
4. `inverse_document_frequency(counts)` : idf = ln((1 + n) / (1 + df)) + 1, où df est le nombre de documents contenant le mot.
5. `tfidf(counts, idf)` : comptages × idf, puis chaque ligne divisée par sa norme euclidienne (une ligne nulle reste nulle).
6. La classe `TfidfFromScratch` avec `fit`, `transform`, `fit_transform` (attributs `vocabulary_` et `idf_`) et `most_similar(query, documents_matrix, k=3)` qui renvoie la liste des `(indice, similarité)` des k documents les plus proches, triés par similarité décroissante.

## Données d'entrée

`datasets/generated/reviews_fr.csv` (800 avis) et de petites phrases dans les tests.

## Sortie attendue

- Même vocabulaire, mêmes IDF et même matrice que `TfidfVectorizer().fit_transform(textes).toarray()` (écart < 10⁻¹⁰).
- Pour la requête « livraison rapide », le document le plus proche contient ces mots.

## Contraintes

- Pas de scikit-learn dans `exercise.py`.
- Un document sans aucun mot connu donne un vecteur nul, sans division par zéro.

## Indices

- **Indice 1** : `(counts > 0).sum(axis=0)` donne le nombre de documents contenant chaque mot.
- **Indice 2** : `np.linalg.norm(w, axis=1, keepdims=True)`.
- **Indice 3** : le motif `\b\w\w+\b` ignore les mots d'une lettre (« l », « d »).

## Vérification

```bash
python -m pytest exercises/advanced/A03_tfidf_from_scratch -q
```

Solution : `solutions/advanced/A03_tfidf_from_scratch/solution.py`.
