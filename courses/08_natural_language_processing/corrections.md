# Corrigés — Module 8 : Traitement automatique du langage

## Leçon 8.1

### Quiz

1. Une unité de texte après découpage. « l'argent » peut donner `["l", "argent"]` (découpage sur l'apostrophe) ou `["l'", "argent"]`, voire `["l'argent"]` selon la règle choisie.
2. Pour que le modèle apprenne « présence d'une adresse web » sans mémoriser chaque adresse, et pour réduire la taille du vocabulaire.
3. Parce qu'il inverse le sens : « pas décevant » deviendrait « décevant ».
4. La racinisation coupe les terminaisons par des règles (« achet ») ; la lemmatisation renvoie la forme de dictionnaire (« acheter ») grâce à un dictionnaire ou un modèle linguistique.
5. Comme tout paramètre appris, il ne doit pas utiliser d'information du jeu de test (fuite) ; et le test doit refléter des textes nouveaux, avec leurs mots inconnus.
6. La fréquence d'un mot est approximativement inversement proportionnelle à son rang : quelques mots sont très fréquents, la plupart sont rares.

### Exercices guidés

```python
import re
import unicodedata
from collections import Counter

import pandas as pd


def strip_accents(text):
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"\b0\d(?:[ .]?\d{2}){4}\b", " <tel> ", text)
    text = re.sub(r"\d+(?:[.,]\d+)?\s*(?:eur|€)", " <montant> ", text)
    text = re.sub(r"\d+(?:[.,]\d+)?", " <num> ", text)
    return strip_accents(text)


def tokenize(text):
    return re.findall(r"<\w+>|\w+", normalize(text))


print(tokenize("Payez 2,49 EUR de frais ou gagnez 500 EUR, code 12345"))

sms = pd.read_csv("datasets/generated/sms_spam.csv")
sms["n_tokens"] = sms["text"].map(lambda t: len(tokenize(t)))
print(sms.groupby("label")["n_tokens"].mean().round(1).to_dict())

counts = Counter(t for text in sms["text"] for t in tokenize(text))
print("share covered by the 50 most frequent tokens:", round(sum(c for _, c in counts.most_common(50)) / sum(counts.values()), 3))


def ngrams(tokens, n):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


spam_bigrams = Counter(b for text in sms.loc[sms["label"] == "spam", "text"] for b in ngrams(tokenize(text), 2))
print(spam_bigrams.most_common(5))
```

```text output
['payez', '<montant>', 'de', 'frais', 'ou', 'gagnez', '<montant>', 'code', '<num>']
{'ham': 10.6, 'spam': 10.8}
share covered by the 50 most frequent tokens: 0.608
[('sur <url>', 65), ('le <tel>', 64), ('de <montant>', 60), ('votre colis', 47), ('colis est', 47)]
```

## Leçon 8.2

### Quiz

1. L'ordre des mots (et donc une partie du sens : négations, relations entre mots).
2. Leur IDF est faible car ils apparaissent dans presque tous les documents.
3. Parce que chaque document ne contient qu'une infime partie du vocabulaire : stocker les zéros gaspillerait énormément de mémoire.
4. Un texte présent à la fois dans l'entraînement et le test est « déjà vu » : le score est trop optimiste. On dédoublonne ou on découpe par groupes de textes identiques.
5. Ils deviennent des variables à part entière (« pas décevant ») au lieu de laisser « décevant » seul porter un sens négatif.
6. Un coefficient positif pousse vers la seconde classe (ordre alphabétique des étiquettes), un négatif vers la première ; plus la valeur absolue est grande, plus le mot ou n-gramme pèse.

### Exercices guidés

```python
import math

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline

print("idf(colis) =", round(math.log((1 + 3) / (1 + 2)) + 1, 3))

sms = pd.read_csv("datasets/generated/sms_spam.csv")
make_model = lambda **kw: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), **kw), LogisticRegression(max_iter=1000))

tr, te = train_test_split(sms, test_size=0.25, stratify=sms["label"], random_state=0)
random_f1 = f1_score(te["label"], make_model().fit(tr["text"], tr["label"]).predict(te["text"]), pos_label="spam")
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(sms, groups=sms["text"]))
train, test = sms.iloc[train_idx], sms.iloc[test_idx]
model = make_model().fit(train["text"], train["label"])
group_f1 = f1_score(test["label"], model.predict(test["text"]), pos_label="spam")
print(f"F1 random split={random_f1:.3f} | group split={group_f1:.3f}")

coefficients = pd.Series(model[-1].coef_[0], index=model[0].get_feature_names_out())
print("classes:", list(model[-1].classes_), "| top spam terms:", list(coefficients.nlargest(10).index))

full = TfidfVectorizer(ngram_range=(1, 2)).fit(train["text"])
pruned = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(train["text"])
print("columns:", len(full.vocabulary_), "->", len(pruned.vocabulary_), "with min_df=2")
```

```text output
idf(colis) = 1.288
F1 random split=0.987 | group split=0.980
classes: ['ham', 'spam'] | top spam terms: ['example', 'http', 'http promo', 'promo', '08', '08 99', 'le 08', 'eur', '99', 'sur http']
columns: 1942 -> 856 with min_df=2
```

Exercice 2 : avec un découpage aléatoire, les nombreux doublons exacts rendent l'évaluation optimiste. Sur ces données, l'écart est faible car les spams synthétiques sont faciles même sans doublons ; sur des données réelles, il peut être important. L'important est de ne pas pouvoir s'en rendre compte sans faire le découpage par groupes.

## Leçon 8.3

### Quiz

1. Le one-hot est un vecteur creux de taille |V| où tous les mots sont équidistants ; l'embedding est un vecteur dense de petite dimension, appris, où des mots de rôle proche sont proches.
2. Comme les autres poids : par rétropropagation de la perte de la tâche ; les lignes de la table correspondant aux mots d'un lot reçoivent un gradient.
3. À indiquer où commence chaque texte dans le tenseur unique d'identifiants concaténés, sans remplissage.
4. Parce que la moyenne (ou la somme) est la même quel que soit l'ordre des vecteurs.
5. Le sens d'un mot se déduit des contextes dans lesquels il apparaît ; des mots aux contextes semblables ont des sens proches.
6. Télécharger les vecteurs une fois (étape en ligne explicite), les stocker localement (par exemple dans `models/`) et les charger depuis ce fichier, sans appel réseau.

### Exercices guidés

```python
print("parameters for |V|=5000, d=64, 3 classes:", 5000 * 64 + (64 * 3 + 3))
```

```text output
parameters for |V|=5000, d=64, 3 classes: 320195
```

Exercices 2 à 4 : reprenez le code de la leçon en modifiant `tokenize` (supprimer les bigrammes), le paramètre `mode` de `nn.EmbeddingBag` et le fichier de données. Résultat mesuré pour l'exercice 2 avec le même découpage et la même graine : **sans** bigrammes, la précision de test est un peu **meilleure** (≈ 0,97 contre ≈ 0,95). Avec seulement 600 avis d'entraînement, les bigrammes ajoutent beaucoup de jetons rares, donc de paramètres, et favorisent le surapprentissage (la leçon montrait déjà une baisse de précision après la 5e époque). L'intuition « plus de contexte, c'est mieux » doit toujours être vérifiée par la mesure. Pour les exercices 3 et 4, mesurez de la même façon : `mode="sum"` donne plus de poids aux textes longs et demande souvent un taux d'apprentissage plus faible.

## Leçon 8.4

### Quiz

1. La requête représente ce que cherche un élément, la clé ce qu'un élément offre pour être trouvé, la valeur l'information qu'il transmet si on le sélectionne.
2. Le produit scalaire de vecteurs de dimension d_k a une variance proportionnelle à d_k : sans division, les scores sont grands, le softmax sature et les gradients s'annulent.
3. Parce que l'attention est insensible à l'ordre : permuter les éléments permute simplement les sorties.
4. À empêcher chaque position de regarder les positions suivantes, indispensable pour un modèle qui génère le texte mot après mot.
5. Découper les documents, les indexer (vecteurs), retrouver les passages proches de la question, générer une réponse à partir de ces passages avec citations, et refuser de répondre si rien de pertinent n'est trouvé.
6. Télécharger une fois les poids en ligne dans un dossier local, puis les charger avec `local_files_only=True` et les variables `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.

### Exercices guidés

```python
import numpy as np
import torch

scores = np.array([1.0, 0.0]) / np.sqrt(2)
weights = np.exp(scores) / np.exp(scores).sum()
print("weights:", weights.round(3))


def attention(Q, K, V):
    s = Q @ K.T / np.sqrt(K.shape[1])
    w = np.exp(s - s.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return w @ V


rng = np.random.default_rng(0)
Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 5))
reference = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(a, dtype=torch.float64).unsqueeze(0) for a in (Q, K, V)))
print("same as PyTorch:", np.allclose(attention(Q, K, V), reference.squeeze(0).numpy()))

mha = torch.nn.MultiheadAttention(embed_dim=32, num_heads=4)
print("MultiheadAttention parameters:", sum(p.numel() for p in mha.parameters()), "= 4 x (32 x 32 + 32)")
```

```text output
weights: [0.67 0.33]
same as PyTorch: True
MultiheadAttention parameters: 4224 = 4 x (32 x 32 + 32)
```

Exercice 3 : la question « Qu'est-ce que la descente de gradient ? » retrouve la fiche D06 (« Descente de gradient ») en premier. Exercice 4 : les projections Q, K, V et la projection de sortie sont chacune une matrice 32 × 32 plus un biais de 32 : 4 × 1056 = 4224 paramètres. Le nombre de têtes ne change pas le total : les 4 têtes se partagent les 32 dimensions (8 chacune).
