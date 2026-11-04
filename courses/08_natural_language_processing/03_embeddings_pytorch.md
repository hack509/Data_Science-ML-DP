# Leçon 8.3 — Embeddings de mots et classifieur neuronal avec PyTorch

## Prérequis

- Leçons 8.1 et 8.2 ; leçons 6.3 et 6.4 (boucles d'entraînement PyTorch).

## Objectifs d'apprentissage

- Comprendre ce qu'est un embedding de mot et pourquoi il remplace le one-hot.
- Utiliser `nn.Embedding` et `nn.EmbeddingBag` pour classer des textes de longueurs variables.
- Entraîner un classifieur de sentiment en PyTorch, de bout en bout, sur CPU.
- Explorer les embeddings appris (voisins les plus proches par similarité cosinus).
- Connaître word2vec et les embeddings pré-entraînés, et leurs conditions d'usage hors ligne.

## Concepts clés

- **One-hot** : un mot = un vecteur de taille |V| avec un seul 1 ; tous les mots sont équidistants.
- **Embedding** (plongement) : un mot = un vecteur dense de petite dimension (16 à 300), appris, où des mots de rôle proche ont des vecteurs proches.
- **Table d'embeddings** : matrice |V| × d ; la ligne i est le vecteur du mot d'identifiant i.
- **Mise en commun** (*pooling*) : combiner les vecteurs des mots d'un texte (moyenne, somme, maximum) en un vecteur de taille fixe.
- **Remplissage** (*padding*) et **décalages** (*offsets*) : deux façons de regrouper des textes de longueurs différentes dans un même lot.
- **Hypothèse distributionnelle** : « on connaît un mot à ses fréquentations » ; base de word2vec.

## Explication détaillée

### Des identifiants aux vecteurs

Après la leçon 8.1, un texte est une liste d'identifiants entiers. Une table d'embeddings `nn.Embedding(|V|, d)` associe à chaque identifiant un vecteur de d nombres, initialisés au hasard puis **appris par rétropropagation** comme les autres poids du réseau. C'est mathématiquement équivalent à multiplier un vecteur one-hot par une matrice, mais en simple lecture de ligne (bien plus rapide).

Pendant l'entraînement d'un classifieur de sentiment, les mots qui jouent le même rôle pour la tâche (« excellent », « parfait ») reçoivent des gradients semblables et leurs vecteurs se rapprochent. Les embeddings encodent donc ce qui est **utile pour la tâche** sur laquelle ils sont entraînés.

### EmbeddingBag : un classifieur simple et rapide

`nn.EmbeddingBag(|V|, d, mode="mean")` calcule directement la moyenne des vecteurs des mots de chaque texte. Pour former un lot, on concatène tous les identifiants dans un seul tenseur et on indique où commence chaque texte (`offsets`) : pas besoin de remplissage. Une couche linéaire produit ensuite les logits. Ce modèle (proche de *fastText*) est un sac de mots appris : il ignore l'ordre, mais ses représentations sont denses. Ajouter les bigrammes au vocabulaire réintroduit un peu d'ordre local, exactement comme en leçon 8.2.

### word2vec et embeddings pré-entraînés

word2vec (2013) apprend des embeddings sans étiquettes, en prédisant les mots du contexte d'un mot (ou l'inverse) sur d'immenses corpus. On obtient des vecteurs généraux où « roi − homme + femme ≈ reine ». Des vecteurs pré-entraînés pour le français existent (fastText, word2vec), mais ce sont des fichiers de plusieurs centaines de Mo à **télécharger** : ce cours ne les exige pas. Si vous en téléchargez un (étape en ligne, une fois), placez-le dans `models/` ; il se charge ensuite hors ligne comme un simple fichier texte ou binaire (voir la leçon 8.4 pour la démarche générale avec des poids locaux).

### Limites

- La moyenne des vecteurs ignore l'ordre (« pas bon » ≈ « bon pas »).
- Un mot a **un seul** vecteur, quel que soit son sens (« avocat » le fruit et le métier). Les Transformers (leçon 8.4) produisent des vecteurs **contextuels**.
- Avec peu de données, un modèle neuronal ne bat pas forcément TF-IDF + régression logistique.

## Intuition mathématique

```text
one-hot(i) · E = E[i]              (E : table |V| × d)
représentation du texte : h = (1/n) Σⱼ E[idⱼ]
logits = h W + b                    W : d × K
similarité de deux mots : cos(E[a], E[b]) = E[a]·E[b] / (‖E[a]‖ ‖E[b]‖)
```

## Applications concrètes

- Classifieurs de texte légers et rapides (modération, tri de tickets, sentiment).
- Recherche sémantique : retrouver des textes proches par similarité de vecteurs.
- Première couche de tous les modèles de langage neuronaux.

## Exemples de code Python

### Exemple 1 — préparer les données : vocabulaire, identifiants, décalages

```python
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

torch.manual_seed(0)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = re.findall(r"\w+", text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]  # unigrams + bigrams


reviews = pd.read_csv("datasets/generated/reviews_fr.csv")
labels = (reviews["sentiment"] == "positif").astype(int).to_numpy()
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(reviews, groups=reviews["text"]))

counts = Counter(token for text in reviews["text"].iloc[train_idx] for token in tokenize(text))
vocab = {"<unk>": 0}
for token, freq in counts.items():
    if freq >= 2:
        vocab[token] = len(vocab)


def encode_batch(texts):
    ids, offsets = [], []
    for text in texts:
        offsets.append(len(ids))
        ids.extend(vocab.get(token, 0) for token in tokenize(text))
    return torch.tensor(ids, dtype=torch.int64), torch.tensor(offsets, dtype=torch.int64)


ids, offsets = encode_batch(["Le casque n'est pas décevant.", "Très lent."])
print("vocabulary size:", len(vocab))
print("ids:", ids.tolist())
print("offsets:", offsets.tolist())
```

### Explication du code

- Le tokeniseur produit les mots et les bigrammes (`pas_decevant`) pour donner au modèle un peu d'ordre local.
- Le vocabulaire est construit sur l'entraînement (découpage par groupes de textes identiques, comme en leçon 8.2).
- `encode_batch` concatène les identifiants de plusieurs textes ; `offsets` donne la position de départ de chaque texte (0, puis la longueur du premier, etc.). Les mots inconnus prennent l'identifiant 0.

### Sortie attendue

```text output
vocabulary size: 424
ids: [38, 129, 130, 55, 76, 2, 131, 132, 133, 134, 0, 14, 226, 0]
offsets: [0, 11]
```

### Exemple 2 — entraîner un classifieur EmbeddingBag

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 32, n_classes: int = 2):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, dim, mode="mean")
        self.output = nn.Linear(dim, n_classes)

    def forward(self, ids, offsets):
        return self.output(self.embedding(ids, offsets))


texts = reviews["text"].to_numpy()
model = TextClassifier(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.CrossEntropyLoss()
rng = np.random.default_rng(0)
test_ids, test_offsets = encode_batch(texts[test_idx])
test_labels = torch.tensor(labels[test_idx])

for epoch in range(1, 16):
    model.train()
    order = rng.permutation(train_idx)
    for start in range(0, len(order), 32):
        batch = order[start : start + 32]
        ids, offsets = encode_batch(texts[batch])
        optimizer.zero_grad()
        loss = loss_fn(model(ids, offsets), torch.tensor(labels[batch]))
        loss.backward()
        optimizer.step()
    if epoch in (1, 5, 15):
        model.eval()
        with torch.no_grad():
            accuracy = (model(test_ids, test_offsets).argmax(1) == test_labels).float().mean().item()
        print(f"epoch {epoch:>2}: last batch loss={loss.item():.3f}  test accuracy={accuracy:.3f}")

n_params = sum(p.numel() for p in model.parameters())
print("parameters:", n_params, "(embedding table:", model.embedding.weight.shape[0], "x", model.embedding.weight.shape[1], ")")
```

### Explication du code

- Le modèle tient en deux couches : la table d'embeddings (avec moyenne intégrée) et une couche linéaire vers 2 classes.
- La boucle forme des lots de 32 avis mélangés à chaque époque ; l'entraînement complet prend quelques secondes sur CPU.
- La précision de test plafonne dès la 5e époque puis baisse légèrement : un début de surapprentissage, que l'arrêt précoce (leçon 6.4) éviterait.
- La précision de test est du même ordre que celle de TF-IDF + régression logistique de la leçon 8.2 : sur un petit corpus, le modèle neuronal n'a pas d'avantage décisif. Son intérêt grandit avec la quantité de données et avec des architectures qui tiennent compte de l'ordre des mots.
- La majorité des paramètres sont dans la table d'embeddings (|V| × 32).

### Sortie attendue

```text output
epoch  1: last batch loss=0.171  test accuracy=0.920
epoch  5: last batch loss=0.045  test accuracy=0.955
epoch 15: last batch loss=0.031  test accuracy=0.949
parameters: 13634 (embedding table: 424 x 32 )
```

### Exemple 3 — explorer les embeddings appris

```python
embeddings = model.embedding.weight.detach()
normalized = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
inverse_vocab = {index: token for token, index in vocab.items()}


def neighbours(word: str, k: int = 4) -> list[str]:
    similarities = normalized @ normalized[vocab[word]]
    best = similarities.argsort(descending=True)[1 : k + 1]
    return [f"{inverse_vocab[i.item()]} ({similarities[i].item():.2f})" for i in best]


for word in ["excellent", "fragile", "pas_lent"]:
    print(f"{word:<14} -> {neighbours(word)}")

direction = model.output.weight[1] - model.output.weight[0]  # towards 'positif'
scores = (embeddings @ direction.detach()).numpy()
order = np.argsort(scores)
print("most negative words:", [inverse_vocab[i] for i in order[:5]])
print("most positive words:", [inverse_vocab[i] for i in order[-5:]])
```

### Explication du code

- On normalise les vecteurs pour que le produit scalaire soit la similarité cosinus, puis on cherche les plus proches voisins d'un mot.
- Les voisins d'« excellent » sont des bigrammes positifs (« et agréable », « est solide »). Ceux de « pas_lent » sont aussi positifs (« je recommande ») : le modèle a appris que ce bigramme est positif alors que « lent » seul est négatif. Les voisins de « fragile », en revanche, sont peu parlants : sur un corpus de 600 avis d'entraînement, beaucoup de vecteurs restent proches de leur initialisation aléatoire et les similarités mesurent en partie des coïncidences. Il ne faut pas sur-interpréter les voisins d'un petit modèle.
- En projetant chaque embedding sur la direction « positif − négatif » de la couche de sortie, on obtient les mots les plus orientés vers chaque classe : c'est l'équivalent des coefficients de la régression logistique de la leçon 8.2.

### Sortie attendue

```text output
excellent      -> ['cher_au (0.53)', 'et_agreable (0.46)', 'est_solide (0.44)', 'vraiment_agreable (0.44)']
fragile        -> ['casque_fragile (0.44)', 'pas_du (0.40)', 'prix (0.39)', 'mais_solide (0.37)']
pas_lent       -> ['je_recommande (0.74)', 'et_de (0.71)', 'excellent_et (0.70)', 'jours_de (0.69)']
most negative words: ['pas_tres', 'pas_excellent', 'qualite_remboursement', 'serait_parfait', 'recommande_pas']
most positive words: ['pas_trop', 'est_excellent', 'et_excellent', 'jours_de', 'et_de']
```

## Erreurs fréquentes

- **Construire le vocabulaire sur tout le corpus** : fuite.
- **Oublier l'identifiant des mots inconnus** : `IndexError` à la prédiction.
- **Décalages incorrects** dans `EmbeddingBag` (doivent commencer à 0 et être croissants).
- **Interpréter les voisins d'un petit modèle comme des synonymes** : ils reflètent la tâche et le corpus, pas le sens général.
- **Croire qu'un embedding moyen comprend la négation** : sans bigrammes, « pas bon » ≈ « bon ».
- **Télécharger implicitement des vecteurs pré-entraînés** dans un script censé fonctionner hors ligne.

## Exercices guidés

1. Calculez le nombre de paramètres du modèle pour un vocabulaire de 5 000 mots, d = 64 et 3 classes.
2. Entraînez le modèle **sans** bigrammes (modifiez `tokenize`). Comment évolue la précision de test ?
3. Remplacez `mode="mean"` par `mode="sum"` puis `mode="max"` et comparez.
4. Appliquez le même modèle au détecteur de spam (`sms_spam.csv`).

Corrigés : [corrections.md](corrections.md), section « Leçon 8.3 ».

## Défis autonomes

- Remplacez la moyenne par un petit réseau récurrent (`nn.GRU`) qui lit les mots dans l'ordre, avec remplissage (`nn.utils.rnn.pad_sequence`). Le gain justifie-t-il le coût sur ce corpus ?
- Visualisez les embeddings en 2D avec une ACP (leçon 5.5), en colorant les mots selon leur score de sentiment.

## Quiz

1. Quelle est la différence entre un vecteur one-hot et un embedding ?
2. Comment les embeddings sont-ils appris dans un classifieur ?
3. À quoi servent les `offsets` de `nn.EmbeddingBag` ?
4. Pourquoi le classifieur par moyenne d'embeddings ignore-t-il l'ordre des mots ?
5. Qu'est-ce que l'hypothèse distributionnelle ?
6. Que faut-il faire pour utiliser des vecteurs pré-entraînés dans un environnement hors ligne ?

Réponses : [corrections.md](corrections.md), section « Leçon 8.3 ».

## Résumé

- Un embedding est une ligne d'une table apprise : un vecteur dense par mot.
- `EmbeddingBag` + `Linear` = un classifieur de texte rapide, entraîné sur CPU en quelques secondes.
- Les embeddings appris regroupent les mots utiles à la tâche ; ils restent non contextuels.
- Les vecteurs pré-entraînés se téléchargent une fois puis se chargent depuis le disque.

## Pour aller plus loin

- Leçon suivante : [Attention, Transformers et RAG](04_attention_transformers_rag.md).
