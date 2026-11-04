# Module 8 — Traitement automatique du langage naturel (NLP)

Ce module apprend à transformer du texte en nombres exploitables par un modèle, puis à classer des textes (spam, sentiment), à représenter des mots par des vecteurs (embeddings), et introduit l'attention, les Transformers et la génération augmentée par récupération (RAG). Les jeux de données sont des SMS et des avis **synthétiques en français** générés localement.

**Aucun modèle pré-entraîné n'est requis.** Les outils classiques qui téléchargent des ressources au premier usage (listes de NLTK, modèles spaCy, modèles Hugging Face) ne sont pas utilisés : stop words, racinisation et tokenisation sont écrits dans les leçons. La leçon 8.4 explique comment utiliser des **poids téléchargés séparément** (étape optionnelle, en ligne une seule fois) puis chargés depuis le disque, hors ligne.

## Connaissances requises

- Module 5 (classification, pipelines, métriques) et module 6 (PyTorch).

## Durée estimée

14 à 18 heures (4 leçons + exercices A03 et A06 + projet 4).

## Objectifs du module

- Nettoyer, normaliser et découper un texte en jetons ; construire un vocabulaire.
- Représenter des textes par sac de mots et TF-IDF ; entraîner des classifieurs de spam et de sentiment.
- Éviter les fuites propres au texte (doublons entre entraînement et test).
- Entraîner des embeddings de mots dans un classifieur PyTorch.
- Comprendre le mécanisme d'attention et l'architecture Transformer.
- Construire un mini-système RAG local (recherche de passages + réponse fondée sur ces passages).

## Leçons

1. [Prétraitement du texte : normalisation, tokenisation, vocabulaire](01_pretraitement_texte.md)
2. [Sac de mots, TF-IDF et classification de textes](02_bow_tfidf_classification.md)
3. [Embeddings de mots et classifieur neuronal avec PyTorch](03_embeddings_pytorch.md)
4. [Attention, Transformers et génération augmentée par récupération (RAG)](04_attention_transformers_rag.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| A03 | TF-IDF implémenté de zéro | avancé |
| A06 | Attention par produit scalaire en NumPy | avancé |

## Jalon pratique

Projet 4 (`projects/04_spam_detection`) : un détecteur de spam complet, évalué sans fuite, avec analyse des erreurs.

## Critères de validation

- Exercices A03 et A06 réussis.
- Score au quiz du module ≥ 70 %.
- Vous savez expliquer ce que mesure un poids TF-IDF et ce que calcule une tête d'attention.
