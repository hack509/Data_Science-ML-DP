# Leçon 2.4 — Probabilités, lois, échantillonnage et inférence

## Prérequis

- Leçon 2.3 (moyenne, écart-type).
- Leçon 1.4 (générateur aléatoire NumPy).

## Objectifs d'apprentissage

- Calculer des probabilités simples, conditionnelles, et appliquer la formule de Bayes.
- Connaître les lois usuelles (Bernoulli, binomiale, uniforme, normale, Poisson) et les simuler.
- Comprendre l'échantillonnage, la loi des grands nombres et le théorème central limite.
- Construire un intervalle de confiance et le lire correctement.
- Mener un test d'hypothèse et interpréter une p-value sans contresens.

## Concepts clés

- **Expérience aléatoire** / **événement** : un tirage dont le résultat est incertain / un ensemble de résultats possibles.
- **Probabilité conditionnelle** P(A | B) : probabilité de A sachant que B est réalisé.
- **Variable aléatoire** (*random variable*) et **loi** (*distribution*) : les valeurs possibles et leurs probabilités.
- **Population / échantillon** (*population / sample*) : l'ensemble étudié / la partie observée.
- **Estimateur** : formule calculée sur l'échantillon pour approcher une quantité de la population (la moyenne d'échantillon estime la moyenne de la population).
- **Erreur standard** (*standard error*) : écart-type d'un estimateur, s / √n pour une moyenne.
- **Intervalle de confiance** (*confidence interval*, CI) : intervalle construit par une méthode qui contient la vraie valeur dans 95 % des échantillons possibles.
- **Hypothèse nulle** H₀, **p-value**, **seuil** α : les ingrédients d'un test statistique.

## Explication détaillée

### Probabilités et probabilités conditionnelles

Une probabilité est un nombre entre 0 et 1. Pour des issues équiprobables, P(A) = nombre d'issues favorables / nombre d'issues possibles. Règles de base : P(non A) = 1 − P(A) ; pour des événements indépendants, P(A et B) = P(A) · P(B).

La probabilité conditionnelle P(A | B) = P(A et B) / P(B) restreint l'univers aux cas où B est vrai. La **formule de Bayes** permet de « retourner » une condition :

```text
P(A | B) = P(B | A) · P(A) / P(B)
```

Exemple de filtre anti-spam : 25 % des SMS sont des spams ; le mot « gratuit » apparaît dans 40 % des spams et 2 % des messages normaux. Sachant que le message contient « gratuit », quelle est la probabilité que ce soit un spam ? P(gratuit) = 0,4 × 0,25 + 0,02 × 0,75 = 0,115, donc P(spam | gratuit) = 0,1 / 0,115 ≈ 0,87. C'est le principe du classifieur **Naive Bayes** (module 5).

### Lois usuelles

| Loi | Décrit | Paramètres | Exemple |
|---|---|---|---|
| Bernoulli | un essai succès/échec | p | un e-mail est un spam ou non |
| binomiale | nombre de succès sur n essais | n, p | nombre de clics sur 100 publicités |
| uniforme | valeurs également probables sur [a, b] | a, b | initialisation aléatoire de poids |
| normale (gaussienne) | erreurs de mesure, sommes de nombreux effets | μ, σ | tailles, bruit d'un capteur |
| Poisson | nombre d'événements rares par intervalle | λ | appels reçus par heure |

`scipy.stats` fournit pour chaque loi la densité ou les probabilités (`pdf`, `pmf`), la fonction de répartition (`cdf`, la probabilité d'être inférieur à une valeur), les quantiles (`ppf`) et des tirages (`rvs`).

### Échantillonnage, loi des grands nombres, théorème central limite

On observe rarement toute une population : on tire un **échantillon**, idéalement **aléatoire** (chaque individu a la même chance d'être tiré) pour éviter les biais. La **loi des grands nombres** dit que la moyenne d'un échantillon se rapproche de la vraie moyenne quand n augmente. Le **théorème central limite** (TCL) ajoute que, pour n assez grand, la distribution des moyennes d'échantillons est approximativement **normale**, même si les données ne le sont pas, avec un écart-type égal à σ/√n. C'est ce qui rend possibles les intervalles de confiance.

Point clé : la précision augmente comme √n. Pour diviser l'erreur par 2, il faut **quatre fois** plus de données.

### Intervalle de confiance

Pour une moyenne, un intervalle de confiance à 95 % est approximativement x̄ ± 1,96 · s/√n (avec la loi de Student pour les petits échantillons). Interprétation correcte : « la méthode utilisée produit des intervalles qui contiennent la vraie moyenne dans 95 % des cas ». Interprétation **incorrecte** : « il y a 95 % de chances que la vraie moyenne soit dans **cet** intervalle » (la vraie moyenne est fixe ; c'est l'intervalle qui est aléatoire). Une alternative sans formule est le **bootstrap** : rééchantillonner les données avec remise des milliers de fois et prendre les percentiles 2,5 % et 97,5 % des moyennes obtenues.

### Test d'hypothèse

Question : les étudiants du groupe C ont-ils une note moyenne différente de ceux du groupe A ?

1. On pose l'**hypothèse nulle** H₀ : « pas de différence entre les moyennes » et l'alternative H₁ : « différence ».
2. On calcule une statistique de test (ici le t de Student de Welch) qui mesure l'écart observé en nombre d'erreurs standard.
3. La **p-value** est la probabilité d'observer un écart au moins aussi grand **si H₀ était vraie**.
4. Si p < α (souvent 0,05), on rejette H₀ ; sinon, on ne la rejette pas (ce qui ne prouve pas qu'elle est vraie !).

Ce qu'une p-value **n'est pas** : la probabilité que H₀ soit vraie, ni la taille de l'effet. Avec beaucoup de données, une différence minuscule et sans intérêt pratique peut être « significative » : rapportez toujours la **taille de l'effet** (la différence de moyennes et son intervalle de confiance). Et si l'on teste 20 hypothèses au seuil de 5 %, on s'attend à environ un faux positif par hasard (problème des **comparaisons multiples**).

## Intuition mathématique

```text
Bayes :                 P(A|B) = P(B|A) P(A) / P(B)
Loi normale :           f(x) = 1/(σ√(2π)) · exp(−(x−μ)² / (2σ²))
Erreur standard :       SE = s / √n
IC 95 % (grand n) :     x̄ ± 1,96 · SE
Statistique de Welch :  t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

## Applications concrètes

- Tests A/B : la nouvelle version d'une page augmente-t-elle vraiment le taux de clic ?
- Comparer deux modèles de ML : la différence de précision est-elle plus grande que le bruit dû au découpage des données ?
- Contrôle qualité : estimer un taux de défaut avec son incertitude.
- Classifieur Naive Bayes et modèles probabilistes en général.

## Exemples de code Python

### Exemple 1 — Bayes et simulation

```python
import numpy as np

p_spam = 0.25
p_word_given_spam, p_word_given_ham = 0.40, 0.02
p_word = p_word_given_spam * p_spam + p_word_given_ham * (1 - p_spam)
print("P(spam | 'gratuit') =", round(p_word_given_spam * p_spam / p_word, 3))

# Check by simulating one million messages.
rng = np.random.default_rng(0)
n = 1_000_000
is_spam = rng.random(n) < p_spam
has_word = np.where(is_spam, rng.random(n) < p_word_given_spam, rng.random(n) < p_word_given_ham)
print("simulated:", round(is_spam[has_word].mean(), 3))
```

### Explication du code

La première partie applique directement la formule de Bayes. La seconde simule un million de messages : pour chaque message, on tire s'il est un spam, puis s'il contient le mot, avec la probabilité qui correspond à sa classe. La proportion de spams **parmi les messages contenant le mot** retrouve la valeur théorique : la simulation est un excellent moyen de vérifier un raisonnement probabiliste.

### Sortie attendue

```text output
P(spam | 'gratuit') = 0.87
simulated: 0.871
```

### Exemple 2 — lois usuelles et théorème central limite

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

print("P(X <= 3) for Binomial(10, 0.5):", round(stats.binom.cdf(3, n=10, p=0.5), 4))
print("P(|Z| < 1.96) for N(0, 1):", round(stats.norm.cdf(1.96) - stats.norm.cdf(-1.96), 4))
print("97.5% quantile of N(0, 1):", round(stats.norm.ppf(0.975), 4))
print("P(X = 0) for Poisson(2):", round(stats.poisson.pmf(0, mu=2), 4))

rng = np.random.default_rng(1)
population = rng.exponential(scale=2.0, size=100_000)  # very skewed data
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].hist(population, bins=60, color="grey")
axes[0].set_title("population (exponential)")
for ax, n in zip(axes[1:], [5, 50]):
    means = rng.choice(population, size=(5000, n)).mean(axis=1)
    ax.hist(means, bins=50)
    ax.set_title(f"means of samples, n={n}")
    print(f"n={n:>2}: std of means={means.std():.3f}  theory sigma/sqrt(n)={population.std() / np.sqrt(n):.3f}")
fig.tight_layout()
Path("outputs/module02").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module02/central_limit_theorem.png", dpi=100)
plt.close(fig)
```

### Explication du code

- `cdf` donne P(X ≤ x), `ppf` est l'inverse (le quantile) : on retrouve le célèbre 1,96.
- La population exponentielle est très asymétrique. Les histogrammes des moyennes d'échantillons (figure `outputs/module02/central_limit_theorem.png`) deviennent de plus en plus symétriques et en cloche quand n augmente : c'est le TCL.
- L'écart-type des moyennes correspond à σ/√n : passer de n = 5 à n = 50 divise l'incertitude par √10 ≈ 3,2.

### Sortie attendue

```text output
P(X <= 3) for Binomial(10, 0.5): 0.1719
P(|Z| < 1.96) for N(0, 1): 0.95
97.5% quantile of N(0, 1): 1.96
P(X = 0) for Poisson(2): 0.1353
n= 5: std of means=0.877  theory sigma/sqrt(n)=0.887
n=50: std of means=0.282  theory sigma/sqrt(n)=0.280
```

### Exemple 3 — intervalle de confiance et test sur les étudiants

```python
import numpy as np
import pandas as pd
from scipy import stats

students = pd.read_csv("datasets/generated/students.csv")
group_a = students.loc[students["group"] == "A", "final_score"].to_numpy()
group_c = students.loc[students["group"] == "C", "final_score"].to_numpy()

mean_a, se_a = group_a.mean(), stats.sem(group_a)
low, high = stats.t.interval(0.95, df=len(group_a) - 1, loc=mean_a, scale=se_a)
print(f"group A: mean={mean_a:.2f}, 95% CI=[{low:.2f}, {high:.2f}], n={len(group_a)}")

rng = np.random.default_rng(0)
boot_means = rng.choice(group_a, size=(10_000, len(group_a))).mean(axis=1)
print("bootstrap 95% CI:", np.percentile(boot_means, [2.5, 97.5]).round(2))

result = stats.ttest_ind(group_c, group_a, equal_var=False)  # Welch's t-test
diff = group_c.mean() - group_a.mean()
print(f"difference C - A = {diff:.2f} points, t = {result.statistic:.2f}, p-value = {result.pvalue:.4f}")
print("reject H0 at 5%:", result.pvalue < 0.05)
```

### Explication du code

- `stats.sem` calcule l'erreur standard s/√n ; `stats.t.interval` construit l'intervalle avec la loi de Student.
- Le bootstrap rééchantillonne le groupe A 10 000 fois avec remise et donne un intervalle très proche, sans formule.
- Le test de Welch (`equal_var=False`) ne suppose pas que les deux groupes ont la même variance : c'est le choix par défaut recommandé.
- Lisez le résultat en deux temps : la **taille de l'effet** (différence en points) puis la p-value. Ici, le groupe C a 0,70 point de plus, mais p ≈ 0,07 > 0,05 : on ne rejette pas H₀.
- Nous connaissons la vérité, car les données sont synthétiques : le script `generate_datasets.py` attribue les groupes **au hasard**, sans aucun effet sur les notes. L'écart observé est donc dû aux seules fluctuations d'échantillonnage. Si nous avions comparé les trois paires de groupes et retenu la plus « significative », nous aurions pu conclure à tort à un effet : c'est le piège des comparaisons multiples.

### Sortie attendue

```text output
group A: mean=9.42, 95% CI=[8.87, 9.96], n=88
bootstrap 95% CI: [8.89 9.94]
difference C - A = 0.70 points, t = 1.80, p-value = 0.0740
reject H0 at 5%: False
```

## Erreurs fréquentes

- **Confondre P(A | B) et P(B | A)** : P(gratuit | spam) ≠ P(spam | gratuit).
- **Lire la p-value comme la probabilité que H₀ soit vraie**.
- **« Non significatif » = « pas d'effet »** : peut-être simplement pas assez de données.
- **Oublier la taille de l'effet** et ne rapporter qu'une p-value.
- **Échantillon non aléatoire** (volontaires, données d'un seul magasin) : aucune formule ne corrige un biais de sélection.
- **Multiplier les tests** jusqu'à trouver p < 0,05 (*p-hacking*).

## Exercices guidés

1. On lance deux dés. Calculez la probabilité d'obtenir une somme de 7, puis la probabilité d'obtenir 7 **sachant** que le premier dé vaut 3. Vérifiez par simulation.
2. Un test médical détecte une maladie (prévalence 1 %) avec une sensibilité de 99 % et un taux de faux positifs de 5 %. Calculez P(malade | test positif).
3. Calculez l'intervalle de confiance à 95 % de la moyenne de `final_score` pour le groupe B.
4. Testez si le taux de réussite (`passed`) diffère entre les groupes A et C (*indice : un t-test sur une variable 0/1 donne une bonne approximation*).

Corrigés : [corrections.md](corrections.md), section « Leçon 2.4 ».

## Défis autonomes

- Simulez 1 000 échantillons de taille 30 d'une loi normale N(10, 2), calculez pour chacun l'IC à 95 % et comptez combien contiennent la vraie moyenne 10. Vous devez trouver environ 950.
- Simulez 1 000 tests t entre deux groupes tirés de la **même** loi. Combien ont p < 0,05 ? Expliquez le lien avec les comparaisons multiples.

## Quiz

1. Quelle est la différence entre P(A | B) et P(A et B) ?
2. Quelle loi modélise le nombre de clients entrant dans un magasin en une heure ?
3. Que dit le théorème central limite ?
4. Comment évolue la largeur d'un intervalle de confiance quand n est multiplié par 4 ?
5. Une p-value de 0,03 signifie-t-elle qu'il y a 3 % de chances que H₀ soit vraie ?
6. Pourquoi faut-il rapporter la taille de l'effet en plus de la p-value ?

Réponses : [corrections.md](corrections.md), section « Leçon 2.4 ».

## Résumé

- Bayes relie P(A | B) et P(B | A) : c'est la base des filtres anti-spam probabilistes.
- Les lois usuelles (Bernoulli, binomiale, normale, Poisson) se manipulent avec `scipy.stats`.
- Les moyennes d'échantillons sont approximativement normales (TCL), avec une erreur standard s/√n.
- Un IC à 95 % décrit la précision d'une estimation ; une p-value mesure la compatibilité des données avec H₀.

## Pour aller plus loin

- Exercice **I05** (`exercises/intermediate/I05_hypothesis_test`) : IC et test, à la main et avec SciPy.
- Module 5, leçon 5.3 : le classifieur Naive Bayes.
- Leçon suivante : [Dérivées, gradients et descente de gradient](05_derivees_gradient.md).
