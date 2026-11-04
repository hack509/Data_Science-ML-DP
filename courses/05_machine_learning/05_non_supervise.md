# Leçon 5.5 — Apprentissage non supervisé : k-means et ACP

## Prérequis

- Leçons 5.1 à 5.4 ; leçon 2.2 (vecteurs, distances) ; leçon 2.3 (variance).

## Objectifs d'apprentissage

- Comprendre l'algorithme k-means et ses hypothèses.
- Choisir le nombre de groupes (méthode du coude, score de silhouette) et interpréter les groupes.
- Comprendre l'analyse en composantes principales (ACP) : variance expliquée, composantes, projection.
- Utiliser l'ACP pour visualiser et compresser des données.
- Connaître les limites de l'évaluation sans étiquettes.

## Concepts clés

- **Clustering** (partitionnement) : regrouper des observations semblables sans étiquettes.
- **Centroïde** (*centroid*) : centre (moyenne) d'un groupe.
- **Inertie** (*inertia*) : somme des distances au carré de chaque point à son centroïde.
- **Score de silhouette** (*silhouette score*) : entre −1 et 1, compare la distance au propre groupe à la distance au groupe voisin.
- **ACP** (*Principal Component Analysis*, PCA) : changement de repère vers des axes orthogonaux classés par variance décroissante.
- **Composante principale** : nouvel axe, combinaison linéaire des variables d'origine.
- **Variance expliquée** (*explained variance ratio*) : part de la variance totale portée par chaque composante.

## Explication détaillée

### Fiche algorithme : k-means

1. **Intuition** : on place k « aimants » au hasard ; chaque client va vers l'aimant le plus proche ; chaque aimant se déplace au centre de ses clients ; on recommence jusqu'à ce que plus rien ne bouge.
2. **Idée mathématique** : minimiser l'inertie Σᵢ ‖xᵢ − μ_c(i)‖² en alternant deux étapes : **affectation** (chaque point au centroïde le plus proche) et **mise à jour** (chaque centroïde devient la moyenne de ses points). Chaque étape fait baisser l'inertie, donc l'algorithme converge, mais vers un minimum **local** : on relance plusieurs fois avec des initialisations différentes (`n_init`) et on garde la meilleure. L'initialisation *k-means++* écarte les centres de départ.
3. **Implémentation** : `KMeans(n_clusters=k, n_init=10, random_state=0)`, attributs `labels_`, `cluster_centers_`, `inertia_`.
4. **Données** : `customers.csv` (500 clients, 6 variables de comportement). Le fichier séparé `customers_true_segments.csv` contient les segments qui ont servi à générer les données : il ne sert **qu'à vérifier** la méthode, jamais à l'entraînement.
5. **Entraînement** : standardisation indispensable (sinon le revenu en k€ et le panier en € dominent les distances).
6. **Évaluation** : méthode du coude (inertie selon k), score de silhouette ; ici, en bonus, comparaison aux vrais segments avec l'indice de Rand ajusté (ARI).
7. **Interprétation** : décrire chaque groupe par les moyennes de ses variables (profil), leur donner un nom métier, vérifier leur taille.
8. **Erreurs fréquentes** : oublier de standardiser ; croire que les groupes « existent » forcément (k-means trouve toujours k groupes, même dans du bruit) ; groupes non sphériques ou de tailles très différentes, mal gérés ; variables catégorielles traitées comme des nombres.
9-10. **Exercices et corrigé** : voir plus bas, exercice I09 (k-means de zéro) et `corrections.md`.

### Fiche algorithme : ACP

1. **Intuition** : photographier un objet 3D sous l'angle qui montre le plus de détails. L'ACP cherche les directions où les données varient le plus.
2. **Idée mathématique** : sur des données centrées (et généralement standardisées), la première composante est le vecteur unitaire u₁ qui maximise la variance des projections X·u₁ ; la deuxième maximise la variance restante en étant orthogonale à la première, etc. Ce sont les vecteurs propres de la matrice de covariance ; la variance portée par chaque composante est la valeur propre associée.
3. **Implémentation** : `PCA(n_components=2)` ; `explained_variance_ratio_`, `components_` (poids des variables dans chaque composante), `transform`, `inverse_transform`.
4. à 6. Mêmes données clients ; on regarde la variance expliquée cumulée.
7. **Interprétation** : une composante se nomme d'après les variables qui y ont les plus grands poids (par exemple « axe âge/fréquentation contre achats en ligne »).
8. **Erreurs fréquentes** : ACP sans standardisation (la variable de plus grande échelle domine) ; interpréter une composante portant peu de variance ; croire que l'ACP sépare forcément les groupes (elle maximise la variance, pas la séparation des classes).

## Intuition mathématique

```text
k-means :  min Σᵢ ‖xᵢ − μ_c(i)‖²        μ_k = moyenne des points du groupe k
silhouette(i) = (b − a) / max(a, b)       a = distance moyenne à son groupe, b = au groupe voisin le plus proche
ACP :      u₁ = argmax_{‖u‖=1} Var(X u)   ⇔   vecteur propre de la covariance de plus grande valeur propre
```

## Applications concrètes

- Segmentation de clientèle pour adapter les offres (projet 3).
- Compression de données, réduction du bruit, visualisation de données en grande dimension.
- Prétraitement : l'ACP peut réduire le nombre de variables avant un autre modèle.

## Exemples de code Python

### Exemple 1 — choisir k : coude et silhouette

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv("datasets/generated/customers.csv")
features = customers.drop(columns="customer_id")
X = StandardScaler().fit_transform(features)

for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"k={k}: inertia={km.inertia_:8.1f}  silhouette={silhouette_score(X, km.labels_):.3f}")
```

### Explication du code

L'inertie diminue toujours quand k augmente (plus de centres = points plus proches d'un centre) : on cherche le « coude » où la baisse ralentit nettement. Le score de silhouette, lui, est maximal pour la partition la plus nette. Ici, le coude se situe vers k = 4 (la baisse d'inertie devient faible au-delà) et la silhouette est maximale pour k = 4, mais de justesse devant k = 3 (0,517 contre 0,511) : les critères orientent, ils ne décident pas seuls. k = 4 correspond au nombre de segments utilisé par le générateur. Sur des données réelles, les critères sont souvent moins tranchés et le choix de k dépend aussi de l'usage métier (combien de segments une équipe marketing peut-elle exploiter ?).

### Sortie attendue

```text output
k=2: inertia=  1934.5  silhouette=0.405
k=3: inertia=   969.1  silhouette=0.511
k=4: inertia=   629.5  silhouette=0.517
k=5: inertia=   542.9  silhouette=0.449
k=6: inertia=   502.4  silhouette=0.402
k=7: inertia=   467.2  silhouette=0.328
```

### Exemple 2 — profils des groupes et comparaison avec la vérité

```python
from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
customers["cluster"] = kmeans.labels_
profile = customers.groupby("cluster").agg(
    n=("customer_id", "size"),
    age=("age", "mean"),
    income=("annual_income_keur", "mean"),
    spending=("spending_score", "mean"),
    visits=("visits_per_month", "mean"),
    online=("online_ratio", "mean"),
    basket=("avg_basket_eur", "mean"),
)
print(profile.round(1).to_string())

truth = pd.read_csv("datasets/generated/customers_true_segments.csv")
merged = customers.merge(truth, on="customer_id")
print(pd.crosstab(merged["cluster"], merged["true_segment"]).to_string())
print("adjusted Rand index:", round(adjusted_rand_score(merged["true_segment"], merged["cluster"]), 3))
```

### Explication du code

- Le **profil** (moyennes par groupe, dans les unités d'origine) permet de nommer les groupes : par exemple un groupe jeune, très connecté, au petit panier ; un groupe aisé au gros panier...
- Les numéros de clusters (0 à 3) sont arbitraires : ils changent d'une exécution à l'autre si l'on change `random_state`.
- Le tableau croisé avec les vrais segments et l'ARI (1 = correspondance parfaite, 0 = hasard) montrent que k-means retrouve presque parfaitement la structure générée. **Avec des données réelles, ces vrais segments n'existent pas** : c'est pour cela que l'interprétation métier est essentielle.

### Sortie attendue

```text output
           n   age  income  spending  visits  online  basket
cluster                                                     
0        144  42.1    38.5      37.6     3.8     0.3    85.2
1        103  45.0    85.5      79.8     2.9     0.5   184.1
2        158  23.6    24.4      68.4     6.1     0.8    35.3
3         95  66.8    30.7      44.2     7.7     0.1    50.2
true_segment  cadres_premium  familles_economes  jeunes_connectes  seniors_fideles
cluster                                                                           
0                          1                142                 0                1
1                        103                  0                 0                0
2                          0                  0               158                0
3                          0                  1                 0               94
adjusted Rand index: 0.985
```

### Exemple 3 — ACP : variance expliquée et visualisation

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from lab.plotting import CATEGORICAL, apply_style

pca = PCA().fit(X)
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("cumulative:", np.cumsum(pca.explained_variance_ratio_).round(3))
loadings = pd.DataFrame(pca.components_[:2].T, index=features.columns, columns=["PC1", "PC2"])
print(loadings.round(2).to_string())

coords = PCA(n_components=2).fit_transform(X)
apply_style()
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for cluster in range(4):
    mask = kmeans.labels_ == cluster
    ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, color=CATEGORICAL[cluster], label=f"groupe {cluster}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} de la variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} de la variance)")
ax.set_title("Les quatre groupes de clients dans le plan de l'ACP")
ax.legend()
Path("outputs/module05").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module05/pca_clusters.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

- Les deux premières composantes portent plus des trois quarts de la variance (77 %), trois composantes 90 % : la représentation en 2D est fidèle mais pas parfaite.
- Les **poids** (*loadings*) indiquent comment chaque variable contribue à chaque composante ; on nomme les axes d'après les poids les plus forts (en valeur absolue).
- La figure `outputs/module05/pca_clusters.png` montre les groupes de k-means bien séparés dans ce plan. Attention : l'ACP n'utilise pas les groupes ; si les groupes se séparent, c'est parce qu'ils diffèrent sur les directions de plus grande variance.

### Sortie attendue

```text output
explained variance ratio: [0.408 0.366 0.127 0.057 0.021 0.02 ]
cumulative: [0.408 0.775 0.902 0.96  0.98  1.   ]
                     PC1   PC2
age                -0.04 -0.61
annual_income_keur  0.57 -0.19
spending_score      0.39  0.38
visits_per_month   -0.43  0.04
online_ratio        0.12  0.64
avg_basket_eur      0.57 -0.21
```

## Erreurs fréquentes

- **Ne pas standardiser** avant k-means ou l'ACP.
- **Choisir k uniquement d'après l'inertie** (elle baisse toujours).
- **Interpréter les numéros de clusters** comme stables ou ordonnés.
- **Évaluer un clustering avec des étiquettes qu'on n'aurait pas en vrai** et s'en contenter.
- **Croire que des groupes trouvés existent forcément** : k-means partitionne même un nuage uniforme.
- **Garder trop peu de composantes** sans regarder la variance expliquée cumulée.

## Exercices guidés

1. Appliquez k-means (k = 4) **sans** standardiser. Calculez l'ARI avec les vrais segments et comparez.
2. Combien de composantes faut-il pour expliquer au moins 90 % de la variance ?
3. Appliquez k-means avec k = 3 sur un nuage de 500 points uniformes dans un carré. Calculez la silhouette. Qu'en concluez-vous ?
4. Reconstruisez les données à partir de 2 composantes (`inverse_transform`) et calculez l'erreur quadratique moyenne de reconstruction.

Corrigés : [corrections.md](corrections.md), section « Leçon 5.5 ».

## Défis autonomes

- Écrivez k-means vous-même (exercice I09) et comparez le résultat à scikit-learn.
- Utilisez `AgglomerativeClustering` ou `DBSCAN` de scikit-learn sur les mêmes données et comparez les groupes.
- Appliquez l'ACP aux images du jeu `shapes.npz` (784 pixels) et visualisez les 2 premières composantes en couleur selon la forme.

## Quiz

1. Quelles sont les deux étapes répétées par k-means ?
2. Pourquoi lancer k-means plusieurs fois (`n_init`) ?
3. Que mesure le score de silhouette ?
4. Que représente la première composante principale ?
5. Pourquoi standardiser avant une ACP ?
6. k-means trouve 4 groupes : prouve-t-il qu'il existe 4 types de clients ?

Réponses : [corrections.md](corrections.md), section « Leçon 5.5 ».

## Résumé

- k-means : affectation au centre le plus proche puis recalcul des centres ; standardiser, relancer, choisir k avec coude et silhouette, interpréter les profils.
- ACP : axes orthogonaux de variance maximale ; variance expliquée et poids pour interpréter ; utile pour visualiser et compresser.
- Sans étiquettes, l'évaluation repose sur des critères internes et sur l'utilité métier.

## Pour aller plus loin

- Exercice **I09** (`exercises/intermediate/I09_kmeans_from_scratch`).
- Projet 3 (`projects/03_customer_segmentation`).
- Leçon suivante : [Pipelines, validation croisée, hyperparamètres et persistance](06_pipelines_validation.md).
