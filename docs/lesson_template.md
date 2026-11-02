# Gabarit d'une leçon

Toutes les leçons du dépôt suivent la même structure. Le test `tests/test_lesson_structure.py` vérifie automatiquement que les sections obligatoires sont présentes.

## Sections

| N° | Section (titre de niveau 2) | Contenu attendu |
|---|---|---|
| 1 | `# Titre de la leçon` | Titre de niveau 1, unique dans le fichier |
| 2 | `## Prérequis` | Leçons ou notions à connaître avant de commencer |
| 3 | `## Objectifs d'apprentissage` | Ce que l'apprenant saura faire à la fin |
| 4 | `## Concepts clés` | Vocabulaire : terme anglais + explication en français |
| 5 | `## Explication détaillée` | Le cœur du cours, progressif, avec des exemples |
| 6 | `## Intuition mathématique` | Optionnelle : formules expliquées pas à pas |
| 7 | `## Applications concrètes` | Où l'on rencontre ce concept dans la vie réelle |
| 8 | `## Exemples de code Python` | Code complet et exécutable |
| 9 | `### Explication du code` | Sous-section après chaque exemple important |
| 10 | `### Sortie attendue` | Bloc ` ```text output ` contenant la vraie sortie |
| 11 | `## Erreurs fréquentes` | Pièges classiques et comment les éviter |
| 12 | `## Exercices guidés` | Exercices avec indications (corrigés dans `corrections.md`) |
| 13 | `## Défis autonomes` | Problèmes ouverts sans indication |
| 14 | `## Quiz` | Questions ; réponses dans `corrections.md` du module |
| 15 | `## Résumé` | Points essentiels à retenir |
| 16 | `## Pour aller plus loin` | Pratique supplémentaire, exercices du dépôt, liens internes |

## Conventions pour le code

- Les blocs marqués ` ```python ` sont **exécutés par les tests**, dans l'ordre, depuis la racine du dépôt, comme si l'apprenant les copiait un par un dans un même script.
- Les blocs marqués ` ```python no-run ` sont illustratifs (code volontairement faux, pseudo-code, téléchargement de poids optionnels...) et ne sont pas exécutés.
- Les blocs ` ```text output ` contiennent la sortie réelle du bloc Python qui précède. Ils sont régénérés par :

```bash
python -m lab.lesson_runner courses/01_python_for_data_science/01_bases_python.md --update-outputs
```

- Les chemins sont relatifs à la racine du dépôt (`datasets/generated/students.csv`). Lancez donc Python ou Jupyter depuis la racine.
- Les figures sont enregistrées dans `outputs/<module>/` (dossier ignoré par Git, recréé à chaque exécution).
- Pour exécuter tout le code d'une leçon : `python -m lab.lesson_runner <chemin_de_la_lecon.md>`.
