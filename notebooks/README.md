# Notebooks

Un notebook par leçon, généré à partir des fichiers Markdown par :

```bash
python scripts/build_notebooks.py
```

| Dossier | Modules |
|---|---|
| `python_basics/` | 1 |
| `data_analysis/` | 2, 3, 4 |
| `machine_learning/` | 5, 10 |
| `deep_learning/` | 6, 7, 8, 9 |

Ouvrez-les dans VS Code (extension Jupyter) ou avec `python -m jupyter lab`. La première cellule place le répertoire de travail à la racine du dépôt, ce qui permet aux chemins des leçons (`datasets/...`, `outputs/...`) de fonctionner. Exécutez les cellules dans l'ordre : certaines réutilisent les variables des précédentes.

Les leçons Markdown sont la source : ne modifiez pas les notebooks pour corriger le cours, ils seraient écrasés à la prochaine génération. Pour vos propres essais, copiez un notebook sous un autre nom.
