# Leçon 7.1 — Perceptron, perceptron multicouche et fonctions d'activation

## Prérequis

- Leçons 6.2 à 6.4.

## Objectifs d'apprentissage

- Connaître l'algorithme du perceptron et sa limite (problème XOR).
- Comprendre le perceptron multicouche (MLP) comme composition de couches denses.
- Comparer les activations ReLU, sigmoïde, tanh et softmax, et savoir où utiliser chacune.
- Comprendre pourquoi l'entropie croisée est la perte naturelle de la classification.

## Concepts clés

- **Perceptron** : neurone à seuil (sortie 0 ou 1), appris par une règle de correction des erreurs (Rosenblatt, 1958).
- **Séparabilité linéaire** (*linear separability*) : deux classes séparables par un hyperplan.
- **Perceptron multicouche** (*Multilayer Perceptron*, MLP) : empilement de couches denses avec activations non linéaires.
- **Couche dense** (*dense layer*, `nn.Linear`) : y = xW + b.
- **ReLU, sigmoïde, tanh, softmax** : fonctions d'activation.
- **Vraisemblance** (*likelihood*) : probabilité des données observées selon le modèle ; l'entropie croisée est l'opposé de son logarithme moyen.
- **Théorème d'approximation universelle** : un MLP à une couche cachée suffisamment large peut approcher toute fonction continue sur un domaine borné.

## Explication détaillée

### Le perceptron

Le perceptron calcule ŷ = 1 si w·x + b > 0, sinon 0. Sa règle d'apprentissage est intuitive : pour chaque exemple mal classé, on déplace la frontière vers lui : w ← w + η(y − ŷ)x et b ← b + η(y − ŷ). Si les classes sont **linéairement séparables**, l'algorithme converge en un nombre fini d'étapes. Sinon, il oscille pour toujours. En 1969, Minsky et Papert soulignent qu'il ne peut pas apprendre le « ou exclusif » (XOR) : aucune droite ne sépare (0,1) et (1,0) de (0,0) et (1,1). Il faudra attendre la rétropropagation, popularisée en 1986, pour entraîner efficacement des réseaux à plusieurs couches.

La différence avec la régression logistique : le perceptron utilise une marche (non dérivable) et ne corrige que les erreurs ; la logistique utilise la sigmoïde (dérivable) et minimise l'entropie croisée, ce qui donne des probabilités et permet la descente de gradient.

### Le perceptron multicouche

Un MLP enchaîne : couche dense → activation → couche dense → activation → ... → couche de sortie. Chaque couche cachée crée de nouvelles variables à partir des précédentes : les premières combinent les entrées brutes, les suivantes combinent ces combinaisons. Le **théorème d'approximation universelle** garantit qu'une seule couche cachée suffit en théorie, mais il peut falloir un nombre énorme de neurones ; en pratique, des réseaux **plus profonds** représentent souvent les mêmes fonctions avec beaucoup moins de paramètres.

### Choisir les activations

| Activation | Formule | Sortie | Usage |
|---|---|---|---|
| ReLU | max(0, z) | [0, +∞[ | couches cachées (défaut) : simple, gradient 1 pour z > 0 |
| sigmoïde | 1/(1 + e⁻ᶻ) | ]0, 1[ | sortie binaire (probabilité) ; à éviter dans les couches cachées profondes |
| tanh | (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ) | ]−1, 1[ | couches cachées de petits réseaux, réseaux récurrents |
| softmax | eᶻᵏ / Σ eᶻʲ | probabilités de somme 1 | sortie multiclasse |
| aucune (identité) | z | ℝ | sortie de régression |

Pourquoi ReLU domine-t-elle les couches cachées ? La sigmoïde et tanh **saturent** : pour |z| grand, leur dérivée est presque nulle (au maximum 0,25 pour la sigmoïde). Dans un réseau profond, la rétropropagation multiplie ces petites dérivées couche après couche, et le gradient des premières couches devient minuscule : c'est le **gradient qui s'évanouit** (*vanishing gradient*, leçon 7.2). La dérivée de ReLU vaut exactement 1 pour z > 0. Son défaut : un neurone dont z reste négatif pour tous les exemples a un gradient nul et ne se « réveille » plus (*dying ReLU*) ; des variantes comme Leaky ReLU (petite pente pour z < 0) ou GELU l'atténuent.

### Pourquoi l'entropie croisée ?

Si le modèle donne la probabilité pᵢ à la vraie classe de l'exemple i, la vraisemblance des données est le produit Π pᵢ. La maximiser revient à minimiser −(1/n) Σ log pᵢ : c'est l'entropie croisée. Elle pénalise énormément une erreur confiante (log de 0,01 ≈ −4,6) et, combinée à softmax ou sigmoïde, donne un gradient simple et bien conditionné (p − y). La MSE sur des probabilités, elle, donne des gradients très faibles quand le modèle se trompe avec assurance.

## Intuition mathématique

```text
perceptron :      w ← w + η (y − ŷ) x
MLP à L couches : h⁽⁰⁾ = x,  h⁽ˡ⁾ = f(h⁽ˡ⁻¹⁾ W⁽ˡ⁾ + b⁽ˡ⁾),  sortie = h⁽ᴸ⁾
σ'(z) = σ(z)(1 − σ(z)) ≤ 0,25      tanh'(z) = 1 − tanh²(z) ≤ 1      ReLU'(z) = 1 si z > 0
entropie croisée = −(1/n) Σ log p(yᵢ | xᵢ) = −log-vraisemblance moyenne
```

## Applications concrètes

- Les MLP restent des modèles de référence pour les données tabulaires et comme « tête » de classification de réseaux plus complexes (CNN au module 9, Transformers au module 8).
- Le choix de l'activation de sortie dépend directement de la tâche : identité (régression), sigmoïde (binaire, multi-étiquettes), softmax (multiclasse).

## Exemples de code Python

### Exemple 1 — l'algorithme du perceptron : succès sur AND, échec sur XOR

```python
import numpy as np


def train_perceptron(X, y, epochs=20, lr=1.0):
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            prediction = int(xi @ w + b > 0)
            if prediction != yi:
                w += lr * (yi - prediction) * xi
                b += lr * (yi - prediction)
                errors += 1
        if errors == 0:
            return w, b, epoch + 1
    return w, b, None


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
for name, y in [("AND", np.array([0, 0, 0, 1])), ("OR", np.array([0, 1, 1, 1])), ("XOR", np.array([0, 1, 1, 0]))]:
    w, b, converged_after = train_perceptron(X, y)
    predictions = (X @ w + b > 0).astype(int)
    status = f"converged after {converged_after} epochs" if converged_after else "did NOT converge in 20 epochs"
    print(f"{name}: {status}; predictions={predictions.tolist()} expected={y.tolist()}")
```

### Explication du code

La règle du perceptron ne modifie les poids que sur les erreurs. AND et OR sont linéairement séparables : l'algorithme trouve une frontière en quelques époques. Pour XOR, il continue de se tromper indéfiniment : aucune droite ne convient.

### Sortie attendue

```text output
AND: converged after 6 epochs; predictions=[0, 0, 0, 1] expected=[0, 0, 0, 1]
OR: converged after 4 epochs; predictions=[0, 1, 1, 1] expected=[0, 1, 1, 1]
XOR: did NOT converge in 20 epochs; predictions=[1, 1, 0, 0] expected=[0, 1, 1, 0]
```

### Exemple 2 — un MLP PyTorch résout XOR ; comparer les activations

```python
import torch
from torch import nn

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
for name, activation in [("ReLU", nn.ReLU()), ("tanh", nn.Tanh()), ("sigmoid", nn.Sigmoid()), ("none (linear)", nn.Identity())]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), activation, nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = (model(X_t) > 0).int().ravel().tolist()
    print(f"{name:<14} final loss={loss.item():.3f} predictions={predictions}")
```

### Explication du code

Avec une couche cachée non linéaire (ReLU, tanh ou sigmoïde), le réseau résout XOR. Sans activation (`nn.Identity`), le réseau est une fonction linéaire et échoue comme le perceptron, quelle que soit la largeur de la couche cachée : la perte reste à log(2) ≈ 0,693, la valeur d'une prédiction constante à 50 %. La sigmoïde converge plus lentement que ReLU et tanh, même sur ce minuscule réseau.

### Sortie attendue

```text output
ReLU           final loss=0.000 predictions=[0, 1, 1, 0]
tanh           final loss=0.000 predictions=[0, 1, 1, 0]
sigmoid        final loss=0.001 predictions=[0, 1, 1, 0]
none (linear)  final loss=0.693 predictions=[0, 0, 0, 1]
```

### Exemple 3 — dérivées des activations et entropie croisée contre MSE

```python
z = torch.linspace(-6, 6, 7, requires_grad=True)
for name, function in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh), ("relu", torch.relu)]:
    z.grad = None
    function(z).sum().backward()
    print(f"{name:<8} derivative at z={[int(v) for v in z.detach().tolist()]}: {[round(g, 3) for g in z.grad.tolist()]}")

# A confident wrong prediction: true class 1, logit = -5 (probability 0.007)
logit = torch.tensor([-5.0], requires_grad=True)
target = torch.tensor([1.0])
bce = nn.functional.binary_cross_entropy_with_logits(logit, target)
bce.backward()
print(f"cross-entropy loss={bce.item():.3f} gradient={logit.grad.item():.3f}")
logit.grad = None
mse = (torch.sigmoid(logit) - target).pow(2).mean()
mse.backward()
print(f"MSE on probability loss={mse.item():.3f} gradient={logit.grad.item():.4f}")
```

### Explication du code

- La dérivée de la sigmoïde ne dépasse jamais 0,25 et tombe à 0,002 pour |z| = 6 ; celle de tanh atteint 1 en 0 mais s'écrase aussi aux extrémités ; celle de ReLU vaut 0 ou 1.
- Pour une prédiction fausse et confiante, l'entropie croisée donne une perte élevée et un gradient fort (≈ −1) : le modèle est vigoureusement corrigé. La MSE sur la probabilité donne un gradient minuscule : le modèle apprendrait à peine de son erreur la plus grave.

### Sortie attendue

```text output
sigmoid  derivative at z=[-6, -4, -2, 0, 2, 4, 6]: [0.002, 0.018, 0.105, 0.25, 0.105, 0.018, 0.002]
tanh     derivative at z=[-6, -4, -2, 0, 2, 4, 6]: [0.0, 0.001, 0.071, 1.0, 0.071, 0.001, 0.0]
relu     derivative at z=[-6, -4, -2, 0, 2, 4, 6]: [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
cross-entropy loss=5.007 gradient=-0.993
MSE on probability loss=0.987 gradient=-0.0132
```

## Erreurs fréquentes

- **Oublier l'activation** entre deux couches `nn.Linear` : le réseau reste linéaire.
- **Sigmoïde dans toutes les couches cachées d'un réseau profond** : gradients qui s'évanouissent.
- **Activation de sortie inadaptée** : ReLU en sortie d'une régression qui peut être négative, softmax pour un problème multi-étiquettes (où plusieurs classes peuvent être vraies : utiliser des sigmoïdes indépendantes).
- **MSE pour une classification** : apprentissage lent sur les erreurs confiantes.
- **Croire que l'approximation universelle garantit qu'on trouvera la fonction** : elle dit qu'elle existe, pas qu'on saura l'apprendre avec les données disponibles.

## Exercices guidés

1. Appliquez à la main deux étapes de la règle du perceptron sur AND en partant de w = (0, 0), b = 0, lr = 1, avec les exemples dans l'ordre.
2. Ajoutez au tableau de l'exemple 2 l'activation `nn.LeakyReLU(0.1)` et `nn.GELU()`.
3. Calculez la dérivée maximale de la sigmoïde et le produit de 10 telles dérivées. Qu'en déduisez-vous pour un réseau de 10 couches sigmoïdes ?
4. Calculez l'entropie croisée de prédictions p = 0,9 ; 0,6 ; 0,1 pour un exemple de classe 1.

Corrigés : [corrections.md](corrections.md), section « Leçon 7.1 ».

## Défis autonomes

- Montrez qu'un MLP sans activation à 3 couches équivaut à une seule couche linéaire : calculez W = W₁W₂W₃ et comparez les sorties.
- Entraînez un MLP de régression (sortie identité, perte MSE) sur `regression.csv` et comparez son R² à la régression linéaire et au k-NN de la leçon 5.2.

## Quiz

1. Pourquoi le perceptron ne peut-il pas apprendre XOR ?
2. Quelle différence entre perceptron et régression logistique ?
3. Quelle activation utiliser pour la sortie d'un problème multiclasse ? D'une régression ?
4. Pourquoi ReLU est-elle préférée à la sigmoïde dans les couches cachées ?
5. Quel est le lien entre entropie croisée et vraisemblance ?
6. Que dit (et ne dit pas) le théorème d'approximation universelle ?

Réponses : [corrections.md](corrections.md), section « Leçon 7.1 ».

## Résumé

- Perceptron : frontière linéaire, ne résout que les problèmes linéairement séparables.
- MLP : couches denses + activations non linéaires ; la profondeur rend les représentations efficaces.
- ReLU dans les couches cachées ; sigmoïde, softmax ou identité en sortie selon la tâche.
- Entropie croisée = log-vraisemblance négative ; gradients forts sur les erreurs confiantes.

## Pour aller plus loin

- Leçon suivante : [Initialisation, courbes d'apprentissage et débogage](02_initialisation_debogage.md).
