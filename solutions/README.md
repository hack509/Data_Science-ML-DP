# Solutions et corrigé d'auto-évaluation

Ce dossier contient une solution de référence par exercice. **Ne l'ouvrez qu'après avoir essayé.** Une solution n'est pas la seule possible : si vos tests passent et que votre code est clair, votre réponse est bonne. Comparez surtout les idées ci-dessous.

Pour vérifier une solution (PowerShell : `$env:LAB_EXERCISE_TARGET="solution"` puis la commande pytest) :

```bash
LAB_EXERCISE_TARGET=solution python -m pytest exercises/beginner/B01_student_averages -q
```

## Corrigé : résultats de référence et points clés

| Exercice | Résultat de référence | Point clé à vérifier dans votre code |
|---|---|---|
| B01 | meilleur étudiant `S0081`, réussite 0.44 | les notes CSV sont converties ; 0 est une vraie note |
| B02 | Nord 153939.0 ; top produit Souris (2591) | arrondi uniquement à la fin ; tri à deux critères |
| B03 | 410 → 379 lignes, CA 64939.08 | doublons retirés avant les filtres ; `raw` non modifié |
| B04 | 124 lignes, 14 clients sans commande | aucun client perdu dans le résumé |
| B05 | identique à NumPy | `ddof` ; quantile par interpolation linéaire |
| B06 | identique à NumPy | compatibilité des formes vérifiée |
| I01 | 379 lignes, Nord en tête | erreurs gérées dans `main`, code de retour 1 |
| I02 | 21 / 15 / 9 valeurs imputées | indicateurs créés avant l'imputation ; médiane par groupe |
| I03 | 10 orphelins, 50 « inconnu » | `validate="one_to_one"`, jointure gauche |
| I04 | MSE ≈ 3.02, égale aux moindres carrés | gradients calculés avant la mise à jour |
| I05 | écart ≈ 0.70, p ≈ 0.074 | ne pas rejeter H₀ ; `ddof=1` ; taille d'effet |
| I06 | 4 PNG, titres calculés | figures fermées ; aucun nombre écrit en dur |
| I07 | R² > 0.9 | tout le prétraitement dans le pipeline |
| I08 | identique à scikit-learn | division par zéro ; égalités de scores dans l'AUC |
| I09 | ARI > 0.95, inertie ≈ 629.5 | affectation vectorisée ; groupe vide conservé |
| A01 | XOR résolu, gradients vérifiés | δ propagé avec la dérivée de ReLU |
| A02 | précision de validation > 0.85 | statistiques d'entraînement seulement ; meilleurs poids restaurés |
| A03 | identique à `TfidfVectorizer` | IDF lissé ; normalisation L2 ; lignes nulles |
| A04 | identique à `F.conv2d` | fenêtres glissantes, pas de boucle sur les pixels |
| A05 | mêmes plis que `KFold` ; fuite > 0.75 contre ≈ 0.5 | `clone` à chaque pli ; sélection dans le pipeline |
| A06 | identique à PyTorch | softmax stable ; mise à l'échelle par √d_head |

Les réponses aux quiz et aux exercices guidés des leçons sont dans `courses/<module>/corrections.md`.
