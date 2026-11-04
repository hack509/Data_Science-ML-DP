# Parcours d'apprentissage

Ce parcours mène d'une connaissance de base de Python à la réalisation de projets complets de Machine Learning et de Deep Learning. Il compte **10 modules, 39 leçons, 21 exercices et 6 projets**, soit environ **130 à 170 heures** de travail. Chaque module se termine par un jalon pratique et des critères de validation : passez au module suivant quand ils sont remplis.

## Vue d'ensemble

```text
 1 Python ──► 2 Maths & stats ──► 3 Analyse ──► 4 Visualisation ──► 5 Machine Learning ──► 10 Projets 1-4, 6
                                                                         │
                                                                         ▼
                                                   6 Deep Learning ──► 7 Réseaux de neurones
                                                         │
                                          ┌──────────────┴──────────────┐
                                          ▼                             ▼
                                       8 NLP                    9 Vision ──► Projet 5
```

Les modules 8 et 9 sont indépendants l'un de l'autre. Les projets 1 et 3 peuvent être commencés dès la fin des modules 4 et 5.

## Tableau de synthèse

| Module | Connaissances requises | Durée | Exercices requis | Jalon pratique |
|---|---|---|---|---|
| 1. Python pour la Data Science | bases de Python | 12-16 h | B01, B02, B03, B04, I01 | script de nettoyage des ventes en ligne de commande |
| 2. Mathématiques et statistiques | module 1 | 15-20 h | B05, B06, I04, I05 | régression linéaire par descente de gradient de zéro |
| 3. Analyse de données | modules 1-2 | 12-15 h | B03, I02, I03 | nettoyage et fusion complets des données étudiants |
| 4. Visualisation | modules 1-3 | 8-10 h | I06 | rapport visuel automatisé |
| 5. Machine Learning | modules 1-4 | 25-30 h | I07, I08, I09, A05 | projet 2 : prix immobiliers |
| 6. Deep Learning | modules 2, 5 | 16-20 h | A01, A02 | réseau multiclasse PyTorch avec arrêt précoce |
| 7. Réseaux de neurones | module 6 | 10-12 h | (A01, A02) + défis | étude comparative d'architectures |
| 8. NLP | modules 5-6 | 14-18 h | A03, A06 | projet 4 : détection de spam |
| 9. Vision par ordinateur | modules 5-6 | 12-15 h | A04 | projet 5 : classification d'images |
| 10. Projets et IA responsable | selon le projet | 6 h + projets | — | projet 6 : pipeline de bout en bout |

## Détail par module

### Module 1 — Python pour la Data Science

- **Connaissances requises** : écrire et lancer un script, variables, `if`, boucles `for`.
- **Durée estimée** : 12 à 16 heures.
- **Objectifs** : structures de données, fonctions, fichiers, exceptions, POO, environnements virtuels, NumPy, pandas, CSV et JSON.
- **Exercices requis** : B01, B02, B03, B04, I01.
- **Jalon** : I01 — script réutilisable qui nettoie `sales_raw.csv` et produit un résumé CSV et JSON.
- **Critères de validation** : tests des exercices réussis ; quiz ≥ 70 % ; savoir expliquer liste vs tableau NumPy et Series vs DataFrame.

### Module 2 — Mathématiques et statistiques

- **Connaissances requises** : module 1, en particulier NumPy.
- **Durée estimée** : 15 à 20 heures.
- **Objectifs** : algèbre et fonctions, vecteurs et matrices, statistiques descriptives, probabilités et inférence, dérivées et descente de gradient.
- **Exercices requis** : B05, B06, I04, I05.
- **Jalon** : I04 — descente de gradient qui retrouve la solution des moindres carrés.
- **Critères** : quiz ≥ 70 % ; savoir expliquer la robustesse de la médiane, la p-value, le rôle du gradient.

### Module 3 — Analyse de données

- **Connaissances requises** : modules 1 et 2.
- **Durée estimée** : 12 à 15 heures.
- **Objectifs** : inspection, nettoyage, valeurs manquantes et aberrantes, création de variables, EDA, agrégations, fusions, contrôles de qualité.
- **Exercices requis** : B03, I02, I03.
- **Jalon** : nettoyage complet de `students_raw.csv` + fusion avec les activités + rapport de qualité.
- **Critères** : quiz ≥ 70 % ; savoir justifier un choix de traitement des valeurs manquantes.

### Module 4 — Visualisation de données

- **Connaissances requises** : modules 1 à 3.
- **Durée estimée** : 8 à 10 heures.
- **Objectifs** : graphiques fondamentaux, distributions, corrélations, choix de la forme, communication honnête et accessible.
- **Exercices requis** : I06.
- **Jalon** : I06 — 4 figures avec titres-constats calculés.
- **Critères** : quiz ≥ 70 % ; repérer trois défauts dans un graphique trompeur.

### Module 5 — Machine Learning

- **Connaissances requises** : modules 1 à 4.
- **Durée estimée** : 25 à 30 heures.
- **Objectifs** : méthodologie (découpages, fuite, biais-variance), régression, classification, arbres et forêts, k-means et ACP, pipelines, validation croisée, hyperparamètres, persistance.
- **Exercices requis** : I07, I08, I09, A05.
- **Jalon** : projet 2 (prix immobiliers).
- **Critères** : quiz ≥ 70 % ; savoir repérer une fuite de données dans un script.

### Module 6 — Deep Learning

- **Connaissances requises** : module 2 (dérivées), module 5 (évaluation) ; PyTorch installé.
- **Durée estimée** : 16 à 20 heures.
- **Objectifs** : tenseurs et autograd, réseau NumPy avec rétropropagation, boucles PyTorch, classification binaire et multiclasse, régularisation, sauvegarde.
- **Exercices requis** : A01, A02.
- **Jalon** : réseau multiclasse avec validation, arrêt précoce et rechargement identique.
- **Critères** : quiz ≥ 70 % ; expliquer `zero_grad`, `backward`, `step`.

### Module 7 — Réseaux de neurones

- **Connaissances requises** : module 6.
- **Durée estimée** : 10 à 12 heures.
- **Objectifs** : perceptron et MLP, activations, initialisation, courbes d'apprentissage, débogage, expérimentation rigoureuse.
- **Exercices** : défis des leçons 7.2 et 7.3 (A01 et A02 en rappel).
- **Jalon** : étude comparative de 4 architectures × 3 graines avec conclusion argumentée.
- **Critères** : quiz ≥ 70 % ; appliquer le test « surapprendre un lot ».

### Module 8 — NLP

- **Connaissances requises** : modules 5 et 6.
- **Durée estimée** : 14 à 18 heures.
- **Objectifs** : prétraitement, sac de mots, TF-IDF, classification de textes, embeddings, attention, Transformers, RAG local.
- **Exercices requis** : A03, A06.
- **Jalon** : projet 4 (détection de spam).
- **Critères** : quiz ≥ 70 % ; expliquer TF-IDF et une tête d'attention.

### Module 9 — Vision par ordinateur

- **Connaissances requises** : modules 5 et 6.
- **Durée estimée** : 12 à 15 heures.
- **Objectifs** : images et prétraitement, convolution et pooling, CNN, augmentation, inférence.
- **Exercices requis** : A04.
- **Jalon** : projet 5 (classification d'images).
- **Critères** : quiz ≥ 70 % ; calculer une taille de sortie de convolution.

### Module 10 — Projets, évaluation, déploiement, IA responsable

- **Connaissances requises** : selon les projets.
- **Durée estimée** : 6 heures de leçons + 4 à 12 heures par projet.
- **Objectifs** : cadrage, évaluation avec incertitude et par sous-groupe, déploiement local, surveillance, équité, fiche modèle.
- **Jalon** : projet 6 (pipeline de bout en bout).
- **Critères** : au moins quatre projets terminés dont le 6 ; quiz ≥ 70 %.

## Comment travailler une leçon

1. Lisez la leçon (dans VS Code, dans l'interface locale ou en PDF).
2. Exécutez le code : copiez les exemples dans un script ou ouvrez le notebook correspondant (`notebooks/`), depuis la racine du dépôt.
3. Faites les exercices guidés **avant** de lire `corrections.md`.
4. Répondez au quiz par écrit, puis comparez avec `corrections.md` et enregistrez votre score :
   `python scripts/progress.py quiz 02_mathematics_and_statistics 5 6`.
5. Marquez la leçon comme terminée (bouton de l'interface ou `python scripts/progress.py lesson <identifiant>`).
6. Notez dans `progress/learning_log.md` ce que vous avez appris et ce qui reste flou.

## Suivi de la progression

```bash
python scripts/progress.py show          # summary
python scripts/progress.py list          # all identifiers
python scripts/run_learning_environment.py --open   # local web interface
```

La progression est stockée dans `progress/progress.json`. Rien n'est marqué automatiquement.
