# Leçon 1.3 — Programmation orientée objet et environnements virtuels

## Prérequis

- Leçons 1.1 et 1.2 (collections, fonctions, modules).

## Objectifs d'apprentissage

- Comprendre ce qu'est une classe, un objet, un attribut et une méthode.
- Écrire une classe simple avec `__init__`, des méthodes et `__repr__`.
- Utiliser les `dataclasses` pour représenter des enregistrements.
- Comprendre l'héritage à travers l'exemple des modèles scikit-learn (`fit` / `predict`).
- Savoir créer, activer et utiliser un environnement virtuel, et pourquoi c'est indispensable.

## Concepts clés

- **Classe** (*class*) : un plan de construction qui décrit des données et des comportements.
- **Objet / instance** (*object / instance*) : une réalisation concrète d'une classe.
- **Attribut** (*attribute*) : une donnée attachée à un objet (`model.coef_`).
- **Méthode** (*method*) : une fonction attachée à un objet (`model.fit(X, y)`).
- **Encapsulation** : regrouper données et comportements, et cacher les détails internes.
- **Héritage** (*inheritance*) : une classe enfant réutilise et spécialise une classe parente.
- **Environnement virtuel** (*virtual environment*) : un dossier contenant un interpréteur Python et ses propres paquets, isolé du reste du système.

## Explication détaillée

### Pourquoi des objets en data science ?

Vous utilisez déjà des objets sans le savoir : une chaîne possède des méthodes (`"abc".upper()`), un `DataFrame` pandas aussi (`df.head()`), et chaque modèle scikit-learn est un objet. Comprendre la POO (programmation orientée objet) permet de lire la documentation de ces bibliothèques et d'écrire ses propres outils réutilisables.

Une classe regroupe un **état** (les attributs) et des **comportements** (les méthodes). Le mot-clé `self` désigne l'objet sur lequel la méthode est appelée : dans `compte.deposer(10)`, Python appelle en réalité `Compte.deposer(compte, 10)`.

La méthode spéciale `__init__` initialise un nouvel objet. `__repr__` définit l'affichage technique d'un objet (pratique pour déboguer). D'autres méthodes spéciales (`__len__`, `__eq__`, `__lt__`...) permettent à vos objets de fonctionner avec `len()`, `==`, `sorted()`.

### Dataclasses

Pour des objets qui servent surtout à **stocker des données** (un étudiant, une transaction), le décorateur `@dataclass` génère automatiquement `__init__`, `__repr__` et `__eq__` à partir des annotations de type. C'est plus court et moins sujet aux erreurs.

### Héritage et l'interface scikit-learn

Tous les modèles scikit-learn suivent la même **interface** : un constructeur qui reçoit les hyperparamètres, une méthode `fit(X, y)` qui apprend et stocke les paramètres appris dans des attributs terminés par `_` (`coef_`, `classes_`), et une méthode `predict(X)`. Grâce à cette convention, on peut remplacer un modèle par un autre en changeant une seule ligne. Dans l'exemple 2, nous écrivons un modèle « naïf » qui respecte cette interface ; vous verrez au module 5 qu'il constitue une **référence** (*baseline*) indispensable.

Avec l'héritage, une classe enfant (`class MedianRegressor(MeanRegressor)`) reçoit toutes les méthodes du parent et n'a besoin de redéfinir que ce qui change. N'en abusez pas : la composition (un objet qui contient d'autres objets) est souvent plus simple.

### Environnements virtuels

Chaque projet dépend de versions précises de bibliothèques. Sans isolation, mettre à jour pandas pour un projet peut casser un autre projet. Un environnement virtuel règle ce problème : c'est un dossier (ici `.venv/`) qui contient son propre `python` et son propre `pip`.

Cycle de vie :

```bash
python -m venv .venv                     # 1. create (offline)
.venv\Scripts\Activate.ps1               # 2. activate on Windows PowerShell
source .venv/bin/activate                # 2. activate on Linux / macOS
python -m pip install -r requirements.txt  # 3. install (ONLINE, once)
python -m pip freeze > versions.txt      # 4. record exact versions
deactivate                               # 5. leave the environment
```

Points importants pour un usage hors ligne :

- Seule l'étape 3 nécessite Internet. Une fois les paquets installés, tout fonctionne sans connexion.
- Pour installer sur une machine **sans** Internet, on peut préparer des fichiers « wheel » sur une machine connectée : `pip download -r requirements.txt -d wheels/`, puis sur la machine hors ligne `pip install --no-index --find-links wheels/ -r requirements.txt` (voir `docs/offline_usage.md`).
- Dans VS Code, choisissez l'interpréteur `.venv` (commande « Python: Select Interpreter ») pour que l'éditeur, les notebooks et les tests utilisent le même environnement.
- `sys.prefix != sys.base_prefix` indique que l'on est dans un environnement virtuel.

## Applications concrètes

- Une classe `DataCleaner` qui mémorise les règles de nettoyage apprises sur les données d'entraînement (médianes, catégories connues) et les applique ensuite aux nouvelles données : c'est exactement le principe des `transformers` scikit-learn (`fit` puis `transform`).
- Une dataclass `Transaction` qui valide ses champs à la création.
- Un environnement virtuel par projet, avec un `requirements.txt` versionné pour que vos collègues reproduisent vos résultats.

## Exemples de code Python

### Exemple 1 — une classe et une dataclass

```python
from dataclasses import dataclass, field


class Gradebook:
    """Collect the scores of one student and compute statistics."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.scores: dict[str, float] = {}

    def add_score(self, subject: str, score: float) -> None:
        if not 0 <= score <= 20:
            raise ValueError(f"invalid score {score} for {subject}")
        self.scores[subject] = score

    def average(self) -> float:
        return sum(self.scores.values()) / len(self.scores)

    def __repr__(self) -> str:
        return f"Gradebook({self.student_id!r}, {len(self.scores)} scores)"


book = Gradebook("S01")
book.add_score("math", 14)
book.add_score("physics", 11.5)
print(book, "-> average", book.average())
try:
    book.add_score("french", 25)
except ValueError as exc:
    print("Refused:", exc)


@dataclass(order=True)
class Transaction:
    amount: float
    product: str
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("amount must be positive")


transactions = [Transaction(39.9, "Clavier"), Transaction(24.9, "Souris", ["promo"])]
print(sorted(transactions)[0])
print(Transaction(24.9, "Souris", ["promo"]) == transactions[1])
```

### Explication du code

- `Gradebook` encapsule les notes : on ne manipule pas le dictionnaire directement, on passe par `add_score` qui **valide** les données.
- `@dataclass(order=True)` génère `__init__`, `__repr__`, `__eq__` et les comparaisons ; le tri se fait d'abord sur `amount`, le premier champ.
- `field(default_factory=list)` crée une nouvelle liste pour chaque objet (même piège que les valeurs par défaut mutables de la leçon 1.2).
- `__post_init__` est appelé après l'initialisation automatique : l'endroit idéal pour valider.

### Sortie attendue

```text output
Gradebook('S01', 2 scores) -> average 12.75
Refused: invalid score 25 for french
Transaction(amount=24.9, product='Souris', tags=['promo'])
True
```

### Exemple 2 — l'interface fit / predict et l'héritage

```python
class MeanRegressor:
    """Baseline model: always predicts the mean of the training targets."""

    def fit(self, X, y):
        self.value_ = sum(y) / len(y)
        return self  # returning self allows model.fit(X, y).predict(X)

    def predict(self, X):
        return [self.value_ for _ in X]


class MedianRegressor(MeanRegressor):
    """Same interface, but predicts the median (robust to outliers)."""

    def fit(self, X, y):
        ordered = sorted(y)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            self.value_ = ordered[middle]
        else:
            self.value_ = (ordered[middle - 1] + ordered[middle]) / 2
        return self


X_train = [[50], [70], [90], [120], [65]]
y_train = [150, 210, 260, 900, 190]  # one luxury house is an outlier
X_new = [[80], [100]]

for model in (MeanRegressor(), MedianRegressor()):
    predictions = model.fit(X_train, y_train).predict(X_new)
    print(f"{type(model).__name__:<16} {predictions}")
print(isinstance(MedianRegressor(), MeanRegressor))
```

### Explication du code

Les deux classes ont exactement la même interface : la boucle les utilise sans savoir laquelle est laquelle (c'est le **polymorphisme**). `MedianRegressor` hérite de `predict` sans le réécrire. La valeur aberrante (900) tire la moyenne vers le haut alors que la médiane reste représentative : retenez cette différence, elle reviendra en statistique (module 2).

### Sortie attendue

```text output
MeanRegressor    [342.0, 342.0]
MedianRegressor  [210, 210]
True
```

### Exemple 3 — inspecter son environnement

```python
import sys
import importlib.metadata as metadata

in_venv = sys.prefix != sys.base_prefix
print("Virtual environment active:", in_venv)
print("Python version:", sys.version.split()[0])
for package in ["numpy", "pandas", "scikit-learn"]:
    try:
        print(f"{package:<13} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package:<13} NOT INSTALLED -> python -m pip install -r requirements.txt")
```

### Explication du code

`importlib.metadata.version` lit la version installée sans importer le paquet. C'est le principe du script `setup/check_environment.py` du dépôt. La sortie dépend de votre machine (versions et présence d'un environnement virtuel).

### Sortie attendue

```text output
Virtual environment active: True
Python version: 3.13.17
numpy         2.5.4
pandas        3.1.0
scikit-learn  1.9.1
```

## Erreurs fréquentes

- **Oublier `self`** dans la définition d'une méthode : `TypeError: add_score() takes 2 positional arguments but 3 were given`.
- **Attributs de classe mutables partagés** : `class A: items = []` est partagé par toutes les instances ; initialisez dans `__init__`.
- **Installer des paquets hors de l'environnement virtuel** : vérifiez avec `python -c "import sys; print(sys.prefix)"`.
- **Versionner le dossier `.venv`** : il est lourd et spécifique à la machine ; versionnez `requirements.txt`.
- **VS Code utilise un autre interpréteur** que le terminal : sélectionnez `.venv` dans la barre d'état.

## Exercices guidés

1. Ajoutez à `Gradebook` une méthode `best_subject()` qui renvoie la matière de meilleure note. *Indice : `max(dico, key=dico.get)`.*
2. Écrivez une dataclass `Student(student_id: str, group: str, scores: dict)` avec une propriété `average` (décorateur `@property`).
3. Écrivez une classe `MostFrequentClassifier` avec `fit(X, y)` qui mémorise la classe la plus fréquente et `predict(X)`.

Corrigés : [corrections.md](corrections.md), section « Leçon 1.3 ».

## Défis autonomes

- Écrivez une classe `StandardScalerFromScratch` avec `fit(values)` (mémorise moyenne et écart-type) et `transform(values)` (renvoie (x − moyenne) / écart-type). Pourquoi faut-il appeler `fit` uniquement sur les données d'entraînement ? (Réponse au module 5, « fuite de données ».)
- Créez un second environnement virtuel dans `outputs/test_env`, installez-y un paquet depuis le cache hors ligne de pip (s'il existe) et comparez `pip list` dans les deux environnements.

## Quiz

1. Quelle est la différence entre une classe et une instance ?
2. À quoi sert `self` ?
3. Que génère automatiquement `@dataclass` ?
4. Pourquoi les modèles scikit-learn ont-ils tous une méthode `fit` et une méthode `predict` ?
5. Quelle étape de la mise en place d'un environnement virtuel nécessite Internet ?
6. Comment savoir, depuis Python, si un environnement virtuel est actif ?

Réponses : [corrections.md](corrections.md), section « Leçon 1.3 ».

## Résumé

- Une classe regroupe données et comportements ; `__init__` initialise, `self` désigne l'objet.
- Les dataclasses simplifient les objets « enregistrements » ; validez dans `__post_init__`.
- L'interface commune `fit` / `predict` de scikit-learn est un exemple de polymorphisme.
- Un environnement virtuel isole les dépendances ; l'installation se fait une fois en ligne, ensuite tout est hors ligne.

## Pour aller plus loin

- Lisez `lab/progress.py` : une vraie classe (`ProgressTracker`) utilisée par ce dépôt.
- `docs/offline_usage.md` : installation sur une machine sans Internet à partir de wheels.
- Leçon suivante : [NumPy : calcul numérique vectorisé](04_numpy.md).
