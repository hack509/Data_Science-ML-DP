# Leçon 8.4 — Attention, Transformers et génération augmentée par récupération (RAG)

## Prérequis

- Leçons 8.2 et 8.3 ; leçon 2.2 (produit scalaire, produit matriciel) ; leçon 6.4 (softmax).

## Objectifs d'apprentissage

- Comprendre le mécanisme d'attention par produit scalaire (requêtes, clés, valeurs).
- Décrire l'architecture d'un bloc Transformer : attention multi-têtes, encodage de position, couche dense, connexions résiduelles, normalisation.
- Manipuler les briques Transformer de PyTorch et comprendre leurs formes de tenseurs.
- Construire un mini-système RAG entièrement local : recherche de passages par TF-IDF, réponse fondée sur ces passages, avec sources.
- Savoir utiliser des poids pré-entraînés téléchargés séparément, puis chargés hors ligne.

## Concepts clés

- **Attention** : chaque élément d'une séquence calcule une moyenne pondérée des autres éléments ; les poids dépendent de leur pertinence.
- **Requête, clé, valeur** (*query, key, value*, Q, K, V) : trois projections linéaires de chaque élément ; la requête « cherche », la clé « s'annonce », la valeur est « l'information transmise ».
- **Auto-attention** (*self-attention*) : Q, K et V viennent de la même séquence.
- **Attention multi-têtes** (*multi-head attention*) : plusieurs attentions en parallèle, chacune spécialisée.
- **Encodage de position** (*positional encoding*) : information d'ordre ajoutée aux embeddings, car l'attention seule ignore l'ordre.
- **Masque causal** (*causal mask*) : interdit à une position de regarder le futur (modèles génératifs).
- **Transformer** : architecture empilant des blocs d'attention et de couches denses (2017).
- **RAG** (*Retrieval-Augmented Generation*) : récupérer des documents pertinents puis générer une réponse fondée sur eux.

## Explication détaillée

### L'idée de l'attention

Dans « Le colis n'est pas arrivé, il est bloqué au dépôt », le mot « il » doit regarder « colis » pour être compris. L'attention donne à chaque mot la possibilité de **consulter tous les autres** et de construire une représentation **contextuelle** : le vecteur de « il » devient un mélange pondéré des vecteurs des mots pertinents. Contrairement aux embeddings de la leçon 8.3, le même mot obtient un vecteur différent selon la phrase.

Calcul pour une séquence de n éléments représentés par une matrice X (n × d) :

1. On calcule Q = XW_Q, K = XW_K, V = XW_V (projections apprises).
2. Le score entre l'élément i et l'élément j est le produit scalaire qᵢ · kⱼ : grand si la requête de i « correspond » à la clé de j.
3. On divise par √d_k (sinon, en grande dimension, les scores deviennent énormes et le softmax sature).
4. Un softmax par ligne transforme les scores en poids positifs de somme 1.
5. La sortie de i est la moyenne des valeurs pondérée par ces poids.

### Le bloc Transformer

Un bloc d'encodeur Transformer enchaîne : attention multi-têtes → addition de l'entrée (connexion résiduelle) → normalisation de couche (*LayerNorm*) → petit réseau dense appliqué à chaque position → résiduel → normalisation. On empile plusieurs blocs. Comme l'attention ne voit pas l'ordre (permuter les mots permute les sorties), on ajoute aux embeddings un **encodage de position**. Les modèles de type BERT (encodeurs) produisent des représentations de texte ; les modèles de type GPT (décodeurs avec masque causal) prédisent le mot suivant et génèrent du texte. Les grands modèles de langage sont des décodeurs Transformer de milliards de paramètres entraînés sur d'immenses corpus : impossible à entraîner sur un ordinateur personnel, mais certains petits modèles peuvent être **exécutés** localement si l'on a téléchargé leurs poids.

### La génération augmentée par récupération (RAG)

Un modèle de langage ne connaît que ce qu'il a vu à l'entraînement, et il peut inventer (*halluciner*). Le RAG ajoute une étape de **recherche** : on retrouve dans une base documentaire les passages les plus pertinents pour la question, puis on demande au modèle de répondre **à partir de ces passages**, en citant ses sources. Les éléments :

1. **Découpage** des documents en passages (*chunks*).
2. **Indexation** : un vecteur par passage (TF-IDF ici ; des embeddings neuronaux dans les systèmes modernes).
3. **Recherche** : les k passages les plus similaires à la question (similarité cosinus).
4. **Génération** : un modèle de langage reçoit la question et les passages. Sans modèle de langage local, on peut produire une réponse **extractive** : renvoyer les phrases les plus pertinentes, avec leur source.
5. **Garde-fou** : si aucun passage n'est assez similaire, répondre « je ne sais pas » plutôt qu'inventer.

Le mini-RAG de cet exemple fonctionne entièrement hors ligne sur `datasets/generated/knowledge_base.json` (14 fiches de cours).

### Utiliser des poids pré-entraînés hors ligne

Les bibliothèques comme `transformers` (Hugging Face) téléchargent par défaut les poids depuis Internet au premier appel. Pour un usage hors ligne, il faut :

1. **Une fois, en ligne** : installer la bibliothèque et télécharger les poids d'un modèle dans un dossier local (par exemple `models/mon-modele/`), en vérifiant sa licence et sa taille (de quelques centaines de Mo à plusieurs Go).
2. **Ensuite, hors ligne** : charger le modèle **depuis le dossier**, en interdisant explicitement tout accès réseau.

```python no-run
# OPTIONAL - not required by this course. Run once ONLINE to download weights:
#   python -m pip install transformers
#   python -c "from transformers import AutoTokenizer, AutoModel; m='<model-name>'; AutoTokenizer.from_pretrained(m).save_pretrained('models/local-model'); AutoModel.from_pretrained(m).save_pretrained('models/local-model')"
# Afterwards, OFFLINE usage only reads the local folder:
import os

os.environ["HF_HUB_OFFLINE"] = "1"        # forbid any download
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/local-model", local_files_only=True)
model = AutoModel.from_pretrained("models/local-model", local_files_only=True)
```

Ce bloc n'est pas exécuté par les tests du cours (il est marqué `no-run`) : la bibliothèque `transformers` et les poids ne font pas partie de l'installation de base.

## Intuition mathématique

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
formes : Q (n × d_k), K (m × d_k), V (m × d_v)  →  poids (n × m), sortie (n × d_v)
multi-têtes : concat(tête₁, ..., tête_h) W_O,  têteᵢ = Attention(XW_Qⁱ, XW_Kⁱ, XW_Vⁱ)
masque causal : score(i, j) = −∞ si j > i  (le softmax donne alors un poids 0)
coût : O(n²) en longueur de séquence (chaque élément regarde tous les autres)
```

## Applications concrètes

- Traduction, résumé, réponse aux questions, assistants conversationnels (Transformers).
- Assistants documentaires d'entreprise fondés sur une base interne (RAG), avec citation des sources.
- Recherche sémantique dans des documents techniques ou juridiques.

## Exemples de code Python

### Exemple 1 — l'attention par produit scalaire en NumPy

```python
import numpy as np


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = scores - scores.max(axis=axis, keepdims=True)  # numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(K.shape[1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=1)
    return weights @ V, weights


words = ["le", "colis", "est", "bloqué", "il", "attend"]
rng = np.random.default_rng(0)
X = rng.normal(size=(6, 8))
X[4] = X[1] + rng.normal(0, 0.1, size=8)  # make "il" similar to "colis" for the demonstration
W_q, W_k, W_v = (np.eye(8) for _ in range(3))  # identity projections keep the example readable
output, weights = attention(X @ W_q, X @ W_k, X @ W_v)
print("attention weights of 'il':", {w: round(float(v), 2) for w, v in zip(words, weights[4])})
print("rows sum to 1:", np.allclose(weights.sum(axis=1), 1), "| output shape:", output.shape)

causal = np.tril(np.ones((6, 6), dtype=bool))
_, causal_weights = attention(X, X, X, mask=causal)
print("causal weights of 'est' (position 2):", causal_weights[2].round(2))
```

### Explication du code

- Les vecteurs sont aléatoires, sauf celui de « il », rendu volontairement proche de « colis ». Avec des projections identité, la requête de « il » ressemble à la clé de « colis » et à la sienne : ces deux mots reçoivent l'essentiel de son attention. Dans un vrai modèle, ce sont les matrices W_Q, W_K, W_V **apprises** qui créent ces correspondances.
- Chaque ligne de poids somme à 1 : la sortie est une moyenne pondérée des valeurs.
- Avec le masque causal (triangle inférieur), la position 2 ne peut regarder que les positions 0, 1 et 2 : les poids des positions futures sont nuls.

### Sortie attendue

```text output
attention weights of 'il': {'le': 0.01, 'colis': 0.48, 'est': 0.02, 'bloqué': 0.04, 'il': 0.46, 'attend': 0.0}
rows sum to 1: True | output shape: (6, 8)
causal weights of 'est' (position 2): [0.17 0.22 0.62 0.   0.   0.  ]
```

### Exemple 2 — les briques Transformer de PyTorch

```python
import math

import torch
from torch import nn

torch.manual_seed(0)
vocab_size, d_model, seq_len, batch = 1000, 32, 12, 4


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        position = torch.arange(seq_len).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * frequencies)
        pe[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("positional_encoding", pe)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, ids):
        h = self.embedding(ids) + self.positional_encoding[: ids.shape[1]]
        h = self.encoder(h)                  # (batch, seq_len, d_model): one contextual vector per token
        return self.classifier(h.mean(dim=1))


model = TinyEncoder().eval()
ids = torch.randint(0, vocab_size, (batch, seq_len))
with torch.no_grad():
    print("logits shape:", tuple(model(ids).shape))
    contextual = model.encoder(model.embedding(ids) + model.positional_encoding)
print("contextual vectors shape:", tuple(contextual.shape))
print("parameters:", sum(p.numel() for p in model.parameters()))

attention_layer = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
x = torch.randn(batch, seq_len, d_model)
_, attn_weights = attention_layer(x, x, x)
print("multi-head attention weights shape (averaged over heads):", tuple(attn_weights.shape))
```

### Explication du code

- L'encodage de position sinusoïdal (celui de l'article original « Attention Is All You Need ») ajoute à chaque position un motif unique de sinus et cosinus de fréquences différentes.
- `nn.TransformerEncoderLayer` contient l'attention multi-têtes, la couche dense, les connexions résiduelles et les normalisations. On en empile 2.
- La sortie de l'encodeur contient **un vecteur contextuel par jeton** ; pour classer le texte, on en fait la moyenne. Ce modèle n'est pas entraîné ici : l'exemple montre la structure et les formes des tenseurs. Les grands modèles suivent le même schéma avec d_model de plusieurs milliers et des dizaines de blocs.

### Sortie attendue

```text output
logits shape: (4, 2)
contextual vectors shape: (4, 12, 32)
parameters: 49154
multi-head attention weights shape (averaged over heads): (4, 12, 12)
```

### Exemple 3 — un mini-RAG local avec citations

```python
import json
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer

with open("datasets/generated/knowledge_base.json", encoding="utf-8") as handle:
    documents = json.load(handle)

# 1. Chunking: one passage per sentence, keeping its source.
passages = []
for doc in documents:
    for sentence in re.split(r"(?<=[.!?])\s+", doc["text"]):
        passages.append({"doc_id": doc["doc_id"], "title": doc["title"], "text": sentence})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


# Stop words matter for retrieval: otherwise "est" or "la" alone can make a passage look relevant.
STOP_WORDS = sorted({
    "a", "au", "aux", "avec", "c", "ce", "ces", "comment", "d", "dans", "de", "des", "du", "elle", "en", "est", "et",
    "il", "l", "la", "le", "les", "leur", "on", "ou", "par", "pas", "pour", "pourquoi", "qu", "que", "quel", "quelle",
    "qui", "se", "sont", "sur", "un", "une",
})

# 2. Indexing with TF-IDF (title added to each passage to give context).
vectorizer = TfidfVectorizer(preprocessor=normalize, stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True)
index = vectorizer.fit_transform([f"{p['title']}. {p['text']}" for p in passages])


def answer(question: str, k: int = 2, threshold: float = 0.12) -> str:
    # 3. Retrieval: cosine similarity (TF-IDF vectors are L2-normalized).
    scores = (index @ vectorizer.transform([question]).T).toarray().ravel()
    best = scores.argsort()[::-1][:k]
    if scores[best[0]] < threshold:
        return "Je ne trouve pas la réponse dans la base documentaire."
    # 4. Extractive "generation": return the retrieved sentences with their sources.
    lines = [f"- {passages[i]['text']} [{passages[i]['doc_id']} : {passages[i]['title']}, score {scores[i]:.2f}]" for i in best if scores[i] >= threshold]
    return "\n".join(lines)


for question in [
    "Comment réduire le surapprentissage ?",
    "Que mesure le rappel d'un classifieur ?",
    "Pourquoi normaliser avant de séparer les données pose problème ?",
    "Quelle est la capitale de l'Australie ?",
]:
    print(f"Q: {question}\n{answer(question)}\n")
```

### Explication du code

- Les 14 fiches sont découpées en phrases ; chaque passage garde l'identifiant et le titre de sa fiche (pour citer la source).
- Une courte liste de mots vides est indispensable ici : sans elle, une question hors sujet partage des mots comme « est » ou « la » avec n'importe quel passage et obtient un score trompeur (essayez de retirer `stop_words=`). C'est le cas où les mots vides sont utiles, contrairement à l'analyse de sentiment (leçon 8.1).
- L'index est une matrice TF-IDF ; la question est vectorisée avec le **même** vectoriseur et comparée à tous les passages par similarité cosinus.
- La « génération » est ici **extractive** : on renvoie les phrases retrouvées avec leurs sources, ce qui garantit qu'aucune information n'est inventée. Dans un système complet, ces passages seraient insérés dans la consigne d'un modèle de langage local (« Réponds uniquement à partir des extraits suivants et cite-les »).
- La question sur l'Australie ne correspond à aucun passage : le seuil de similarité déclenche une réponse « je ne trouve pas », le comportement souhaité.
- Pour la première question, le second extrait (sur l'ACP) n'est retenu qu'à cause du mot « réduire » : la recherche lexicale est grossière, d'où l'intérêt d'afficher les scores et les sources pour que le lecteur juge.
- Limite de TF-IDF : il faut des mots communs entre question et passage ; des embeddings neuronaux retrouveraient aussi les paraphrases (« apprendre par cœur » pour « surapprentissage »).

### Sortie attendue

```text output
Q: Comment réduire le surapprentissage ?
- La régularisation, plus de données ou un modèle plus simple aident à le réduire. [D03 : Surapprentissage, score 0.31]
- Elle sert à réduire la dimension et à visualiser des données. [D08 : ACP, score 0.21]

Q: Que mesure le rappel d'un classifieur ?
- Le rappel mesure la part de positifs réels retrouvés. [D13 : Précision et rappel, score 0.43]
- La précision mesure la part de vrais positifs parmi les prédictions positives. [D13 : Précision et rappel, score 0.28]

Q: Pourquoi normaliser avant de séparer les données pose problème ?
- La fuite de données (data leakage) arrive quand une information du jeu de test influence l'entraînement, par exemple en normalisant avant de séparer les données. [D05 : Fuite de données, score 0.39]

Q: Quelle est la capitale de l'Australie ?
Je ne trouve pas la réponse dans la base documentaire.
```

## Erreurs fréquentes

- **Oublier la division par √d_k** : softmax saturé, gradients faibles.
- **Oublier l'encodage de position** : le modèle ne distingue pas « le chien mord l'homme » de « l'homme mord le chien ».
- **Confondre la forme `batch_first`** des modules PyTorch : `(batch, séquence, d)` ou `(séquence, batch, d)`.
- **RAG sans seuil ni sources** : le système répond avec assurance même quand la base ne contient pas la réponse.
- **Indexer avec un vectoriseur et interroger avec un autre** (prétraitement différent).
- **Supposer qu'une bibliothèque fonctionne hors ligne** sans l'avoir testée réseau coupé (`setup/verify_offline.py` le fait pour ce cours).

## Exercices guidés

1. Calculez à la main les poids d'attention d'une requête q = (1, 0) sur les clés k₁ = (1, 0), k₂ = (0, 1), avec d_k = 2.
2. Vérifiez que la fonction `attention` de l'exemple 1 donne le même résultat que `torch.nn.functional.scaled_dot_product_attention`.
3. Ajoutez la question « Qu'est-ce que la descente de gradient ? » au mini-RAG et examinez les sources.
4. Comptez les paramètres d'un `nn.MultiheadAttention(embed_dim=32, num_heads=4)` et expliquez le résultat.

Corrigés : [corrections.md](corrections.md), section « Leçon 8.4 ».

## Défis autonomes

- Exercice **A06** : attention multi-têtes et masque causal en NumPy.
- Entraînez le `TinyEncoder` sur les avis `reviews_fr.csv` (avec remplissage à une longueur fixe et un vocabulaire de la leçon 8.3). Comparez-le à `EmbeddingBag`.
- Évaluez le mini-RAG : écrivez 10 questions avec la fiche attendue et mesurez la proportion de bonnes fiches en première position (*precision@1*).

## Quiz

1. Que représentent Q, K et V ?
2. Pourquoi divise-t-on les scores par √d_k ?
3. Pourquoi un Transformer a-t-il besoin d'un encodage de position ?
4. À quoi sert un masque causal ?
5. Quelles sont les étapes d'un système RAG ?
6. Comment utiliser un modèle pré-entraînés sans accès à Internet ?

Réponses : [corrections.md](corrections.md), section « Leçon 8.4 ».

## Résumé

- Attention = softmax(QKᵀ/√d_k)V : chaque élément devient une moyenne pondérée et contextuelle des autres.
- Un Transformer empile attention multi-têtes, couches denses, résiduels et normalisations, avec un encodage de position.
- RAG = découper, indexer, retrouver, répondre à partir des sources, savoir dire « je ne sais pas ».
- Les poids pré-entraînés se téléchargent une fois, puis se chargent depuis le disque en mode hors ligne strict.

## Pour aller plus loin

- Exercice **A06** (`exercises/advanced/A06_attention`).
- Module 10, leçon 10.2 : IA responsable (hallucinations, sources, limites).
