# Glossaire anglais — français

Les termes techniques apparaissent en anglais dans le code et la documentation des bibliothèques. Ce glossaire donne l'équivalent français et une explication courte, avec la leçon où le terme est introduit.

| Terme anglais | Français | Explication | Leçon |
|---|---|---|---|
| accuracy | exactitude | proportion de prédictions correctes | 5.3 |
| activation function | fonction d'activation | fonction non linéaire appliquée à la sortie d'un neurone | 6.2 |
| attention | attention | moyenne pondérée des éléments d'une séquence selon leur pertinence | 8.4 |
| augmentation (data) | augmentation de données | variantes artificielles des exemples d'entraînement | 9.3 |
| autograd | différentiation automatique | calcul automatique des gradients par PyTorch | 6.1 |
| backpropagation | rétropropagation | calcul des gradients couche par couche (règle de la chaîne) | 6.2 |
| bag of words | sac de mots | texte représenté par des comptages de mots | 8.2 |
| baseline | référence | modèle simple que tout modèle doit battre | 5.1 |
| batch | lot | sous-ensemble d'exemples traité en une fois | 6.3 |
| batch normalization | normalisation par lots | standardisation des activations sur chaque lot | 6.4 |
| bias (statistics) | biais | erreur systématique d'un modèle trop simple | 5.1 |
| bootstrap | bootstrap | rééchantillonnage avec remise pour estimer une incertitude | 2.4, 10.1 |
| broadcasting | diffusion | combinaison de tableaux de formes compatibles | 1.4 |
| channel | canal | composante d'une image (rouge, vert, bleu) ou carte d'une couche | 9.1 |
| class imbalance | déséquilibre de classes | une classe beaucoup plus fréquente que les autres | 5.3 |
| clustering | partitionnement | regroupement d'observations sans étiquettes | 5.5 |
| confidence interval | intervalle de confiance | plage plausible pour une quantité estimée | 2.4 |
| confusion matrix | matrice de confusion | tableau réel × prédit des classes | 5.3 |
| convolution | convolution | filtre qui glisse sur l'image | 9.2 |
| cross-entropy | entropie croisée | perte de classification (log-vraisemblance négative) | 5.3, 7.1 |
| cross-validation | validation croisée | évaluation sur k plis successifs | 5.6 |
| data drift | dérive des données | changement de la distribution des entrées | 10.2 |
| data leakage | fuite de données | information de l'évaluation utilisée à l'entraînement | 5.1 |
| DataFrame | table de données | tableau pandas à colonnes typées | 1.5 |
| decision boundary | frontière de décision | limite entre les zones de chaque classe | 5.3 |
| decision tree | arbre de décision | suite de questions qui découpe les données | 5.4 |
| deep learning | apprentissage profond | réseaux de neurones à nombreuses couches | 5.1 |
| dropout | abandon (dropout) | désactivation aléatoire de neurones à l'entraînement | 6.4 |
| early stopping | arrêt précoce | arrêt quand la validation ne s'améliore plus | 6.4 |
| embedding | plongement | vecteur dense appris représentant un mot | 8.3 |
| epoch | époque | un passage complet sur les données d'entraînement | 6.3 |
| feature | variable / caractéristique | entrée d'un modèle | 5.1 |
| feature engineering | ingénierie des variables | création de variables informatives | 3.2 |
| feature map | carte de caractéristiques | sortie d'un filtre de convolution | 9.2 |
| fine-tuning | ajustement fin | réentraînement partiel d'un modèle pré-entraîné | 8.4 |
| gradient | gradient | vecteur des dérivées partielles | 2.5 |
| gradient descent | descente de gradient | optimisation par petits pas opposés au gradient | 2.5 |
| ground truth | vérité terrain | vraie valeur de la cible | 5.1 |
| hallucination | hallucination | réponse inventée par un modèle de langage | 8.4 |
| hyperparameter | hyperparamètre | réglage choisi avant l'entraînement | 5.1 |
| imputation | imputation | remplacement de valeurs manquantes | 3.1 |
| inference | inférence | utilisation d'un modèle entraîné pour prédire | 9.3 |
| learning curve | courbe d'apprentissage | performance selon les époques ou la taille des données | 7.2 |
| learning rate | taux d'apprentissage | taille des pas de la descente de gradient | 2.5 |
| logit | logit | score avant la sigmoïde ou le softmax | 6.3 |
| loss function | fonction de perte | mesure de l'erreur à minimiser | 2.5 |
| machine learning | apprentissage automatique | apprentissage de règles à partir d'exemples | 5.1 |
| mean absolute error (MAE) | erreur absolue moyenne | moyenne des écarts absolus | 5.2 |
| mean squared error (MSE) | erreur quadratique moyenne | moyenne des écarts au carré | 5.2 |
| model card | fiche modèle | documentation standardisée d'un modèle | 10.2 |
| one-hot encoding | encodage one-hot | une colonne binaire par catégorie | 3.2 |
| optimizer | optimiseur | algorithme de mise à jour des poids (SGD, Adam) | 6.3 |
| outlier | valeur aberrante | observation très éloignée des autres | 3.2 |
| overfitting | surapprentissage | mémorisation de l'entraînement, mauvaise généralisation | 5.1 |
| padding | remplissage | bordure ou jetons ajoutés pour égaliser des tailles | 8.1, 9.2 |
| pipeline | pipeline | enchaînement prétraitement + modèle | 5.6 |
| pooling | sous-échantillonnage (pooling) | réduction de taille d'une carte de caractéristiques | 9.2 |
| precision | précision | part de vrais positifs parmi les prédits positifs | 5.3 |
| principal component analysis (PCA) | analyse en composantes principales (ACP) | projection sur les directions de plus grande variance | 5.5 |
| random forest | forêt aléatoire | ensemble d'arbres décorrélés | 5.4 |
| recall | rappel | part des positifs réels retrouvés | 5.3 |
| regularization | régularisation | technique qui limite le surapprentissage | 5.2, 6.4 |
| retrieval-augmented generation (RAG) | génération augmentée par récupération | réponse fondée sur des documents retrouvés | 8.4 |
| ROC curve / AUC | courbe ROC / aire sous la courbe | performance de classement pour tous les seuils | 5.3 |
| seed | graine | valeur qui rend l'aléatoire reproductible | 1.4 |
| softmax | softmax | transformation de scores en probabilités de somme 1 | 6.4 |
| standardization | standardisation | moyenne 0, écart-type 1 | 1.4 |
| stemming / lemmatization | racinisation / lemmatisation | regroupement des formes d'un mot | 8.1 |
| stop words | mots vides | mots très fréquents peu informatifs | 8.1 |
| supervised learning | apprentissage supervisé | apprentissage avec étiquettes | 5.1 |
| tensor | tenseur | tableau multidimensionnel de PyTorch | 6.1 |
| test set | jeu de test | données utilisées une seule fois pour l'évaluation finale | 5.1 |
| TF-IDF | TF-IDF | pondération fréquence × rareté | 8.2 |
| threshold | seuil | valeur de décision sur une probabilité | 5.3 |
| token / tokenization | jeton / tokenisation | unité de texte / découpage du texte | 8.1 |
| training set | jeu d'entraînement | données d'apprentissage | 5.1 |
| transformer | Transformer | architecture fondée sur l'attention | 8.4 |
| underfitting | sous-apprentissage | modèle trop simple | 5.1 |
| unsupervised learning | apprentissage non supervisé | recherche de structure sans étiquettes | 5.5 |
| validation set | jeu de validation | données pour choisir modèles et réglages | 5.1 |
| vanishing gradient | gradient évanescent | gradient qui devient minuscule dans les premières couches | 7.2 |
| variance (model) | variance | sensibilité d'un modèle aux données d'entraînement | 5.1 |
| vectorization | vectorisation | calcul sur des tableaux entiers sans boucle | 1.4 |
| weight decay | pénalité L2 sur les poids | régularisation qui garde les poids petits | 6.4 |
| weight initialization | initialisation des poids | valeurs des poids avant l'entraînement | 7.2 |
