# Veille technique — NLP / Text Mining

Cadre de travail pour l'étape 1 (recherche et cartographie). À compléter par chaque membre avant la réunion de mise à niveau.

Pour chaque notion/méthode, chercher à répondre à ces questions :
- Principe général (en une phrase de vulgarisation)
- Avantages
- Limites / inconvénients
- Pertinence pour ce projet (tweets : textes courts, bruités, hashtags, abréviations, emojis)
- Sources consultées

**Ressources de cours ajoutées (29/07/2026) :** deux documents fournis par l'école sont maintenant dans le repo et alimentent les sections ci-dessous :
- [`B2_Eye-of-Emergency_Ressources.md.pdf`](consignes/B2_Eye-of-Emergency_Ressources.md.pdf) — liste de ressources classée par question du sujet (repères 🔴 incontournable / 🟡 utile / ⚪ pour aller plus loin).
- [`B2_HowTo_Du-tweet-brut-au-vecteur.pdf`](consignes/B2_HowTo_Du-tweet-brut-au-vecteur.pdf) — support d'atelier (1h) qui déroule le pipeline nettoyer → tokeniser → normaliser → vectoriser → vérifier sur un tweet réel, avec du code.

Ces documents donnent des sources et des points de vigilance factuels (repris ci-dessous) — mais **pas** les réponses "avantages / limites / pertinence pour ce projet" : ces champs restent à rédiger par toi, avec ton propre raisonnement, pour que la veille garde sa valeur d'exercice.

## 1. Prétraitement du texte
- Tokenisation
- Nettoyage (stopwords, ponctuation, casse)
- Normalisation (stemming vs lemmatisation)
- Gestion des spécificités Twitter (hashtags, mentions, URLs, emojis)

**Sources consultées :**
- Python `re` (doc officielle + HOWTO regex), [regex101.com](https://regex101.com)
- Stanford IR Book — [Dropping common terms: stop words](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html)
- Stanford IR Book — [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
- NLTK Book ch.3 (normalisation, ponctuation, Unicode), ch.5 (POS tagging)
- L'algorithme de Porter (Martin Porter)
- [`B2_HowTo_Du-tweet-brut-au-vecteur.pdf`](consignes/B2_HowTo_Du-tweet-brut-au-vecteur.pdf), étapes 1 à 3 (Nettoyer / Tokeniser / Normaliser)

**Points de vigilance (issus du cours) :**
- `not`, `no`, `nor` sont dans la stop-list NLTK par défaut : les supprimer rend « not a real fire » identique à « a real fire » — deux classes opposées deviennent indiscernables. À exclure explicitement de la stop-list.
- Mettre en minuscules (`.lower()`) *avant* les regex de nettoyage simplifie leur écriture.
- Hashtag → retirer le `#` mais garder le mot ; mention `@pseudo` → supprimer ou remplacer par `<USER>` (peut devenir une feature à part : « mentionne un compte de secours ») ; emoji → `emoji.demojize()` plutôt que suppression pure (porte du sentiment).
- Compter les `!` et la proportion de MAJUSCULES peuvent devenir des features numériques plutôt que d'être simplement jetés (un tweet paniqué en écrit plus).
- Le `WordNetLemmatizer` de NLTK a besoin du POS tag (`pos='v'`), sinon il suppose "nom" et laisse les verbes intacts.
- Stanford IR Book : le gain du stemming/lemmatisation en anglais est faible (0 à 5 %) — à vérifier empiriquement (GridSearch) plutôt qu'à trancher au feeling.

*Principe général / avantages / limites / pertinence pour le projet : (à compléter)*

## 2. Représentation du texte
- Bag of Words / TF-IDF
- Word embeddings (Word2Vec, GloVe, FastText)
- Embeddings contextuels (BERT et dérivés)

**Sources consultées :**
- scikit-learn — [Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- Stanford IR Book — [tf-idf weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html)
- `CountVectorizer` / `TfidfVectorizer` (doc API scikit-learn)
- scikit-learn — tutoriel *Working With Text Data*
- [`B2_HowTo_Du-tweet-brut-au-vecteur.pdf`](consignes/B2_HowTo_Du-tweet-brut-au-vecteur.pdf), étape 4 (Vectoriser)

**Points de vigilance (issus du cours) :**
- La formule d'idf de scikit-learn diffère de celle des cours théoriques : `idf(t) = ln((1+n)/(1+df(t))) + 1` (lissage activé par défaut) → un mot présent dans tous les documents obtient `idf = 1`, pas `0`.
- Passer en bigrammes (`ngram_range=(1,2)`) fait exploser le nombre de colonnes (~20 000 → 200 000+ sur ce corpus) → `min_df` pour couper les bigrammes vus une seule fois (hapax).
- `min_df` / `max_df` sont, d'après le support de cours, les deux paramètres qui font le plus bouger le F1 — donc à inclure dans le GridSearch (question 8 du sujet).
- Aucune ressource sur les embeddings (Word2Vec, GloVe, BERT...) n'a été fournie dans ce lot — section encore non couverte, veille à poursuivre si le projet s'y prête.

*Principe général / avantages / limites / pertinence pour le projet : (à compléter)*

## 3. Méthodes de classification classiques
- Naive Bayes
- Régression logistique
- SVM
- Random Forest / Gradient Boosting

**Sources consultées :**
- scikit-learn — [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) (bagging et boosting sur la même page)
- Régression logistique, Arbres de décision, SVM (doc scikit-learn)
- XGBoost — *Introduction to Boosted Trees*, XGBoost — API scikit-learn

**Points de vigilance (issus du cours) :**
- `LinearSVC` est très performant sur du texte creux — à préférer à `SVC` pour le temps de calcul.
- Phrase-repère pour l'oral : bagging = plusieurs modèles entraînés en parallèle sur des échantillons différents, on moyenne (réduit la variance) ; boosting = modèles entraînés en séquence, chacun corrige les erreurs du précédent (réduit le biais).
- Le sujet demande aussi de coder son propre Decision Tree (Gini/entropie, récursion, conditions d'arrêt `max_depth`/`min_samples_split`) — sur une matrice TF-IDF pleine (~20 000 colonnes) l'implémentation maison sera très lente : le tester sur un sous-ensemble de features (`max_features` du vectoriseur ≈ 500) est une réponse valable, à condition d'expliquer la limite.

*Principe général / avantages / limites / pertinence pour le projet : (à compléter)*

## 4. Approches deep learning
- RNN / LSTM / GRU
- CNN pour le texte
- Transformers / modèles pré-entraînés (BERT, RoBERTa, DistilBERT...)

Aucune ressource de cours reçue ne couvre cette section pour l'instant (les documents ajoutés se concentrent sur le pipeline classique : nettoyage → TF-IDF → modèles classiques/ensemblistes). À rechercher si le projet doit aller jusque-là.

*(à compléter)*

## 5. Spécificités de la détection de fake news
- Signaux textuels (style, sentiment, sensationnalisme)
- Signaux contextuels (métadonnées, propagation, auteur)
- Jeux de données et benchmarks de référence

**Sources consultées :**
- [Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) : la compétition d'origine, leaderboard permanent (un bon score « fait maison » tourne autour de 0.79–0.81 en F1).
- Description des données Kaggle : colonnes `id`, `keyword`, `location`, `text`, `target` ; 7 613 tweets en train, 3 263 en test. Le test Kaggle n'a pas de `target` → il faut découper le train pour évaluer localement (sinon impossible de calculer un F1).
- Une solution publique commentée (elmezianech) — repère ⚪ à consulter *après* avoir essayé, pas avant.

*Principe général / avantages / limites / pertinence pour le projet : (à compléter)*

## 6. Évaluation
- Métriques (accuracy, precision, recall, F1, AUC)
- Enjeux du déséquilibre de classes
- Biais et limites de l'évaluation

**Sources consultées :**
- scikit-learn — Precision, recall et F-measure
- `classification_report`, `ConfusionMatrixDisplay` (doc API scikit-learn)

**Points de vigilance (issus du cours) :**
- Classes déséquilibrées (~43 % de `target = 1` d'après la ressource de cours) → c'est pourquoi le sujet impose le F1 et non l'accuracy. Ne pas oublier `scoring='f1'` dans `GridSearchCV` (le défaut est l'accuracy).
- Interprétation métier : dans un contexte d'alerte, un faux négatif (une vraie catastrophe classée « non-catastrophe ») est bien plus grave qu'un faux positif → argument pour surveiller le rappel autant que le F1.

*Principe général / avantages / limites / pertinence pour le projet : (à compléter)*

## 7. Pièges classiques à éviter (checklist issue du cours)
1. Supprimer les négations avec les stop-words → deux classes deviennent indiscernables.
2. Vectoriser avant le split train/test → fuite de données, score gonflé et invalide.
3. Optimiser l'accuracy au lieu du F1 alors que le sujet impose le F1.
4. Un wordcloud global au lieu d'un wordcloud par classe → aucune information exploitable.
5. Ne pas refaire l'EDA après le prétraitement (cf. [`eda_texte_nlp.md`](eda_texte_nlp.md), étape 5 du HowTo « Vérifier »).
6. Choisir stemming ou lemmatisation au feeling au lieu de trancher par un chiffre du GridSearch.

---

## Étape 3 — Scénarios de démonstration

Une fois la cartographie avancée, définir 2-3 scénarios concrets sur `data/raw/train_tweets.csv` pour comparer des méthodes entre elles. Pour chaque scénario :

**Scénario n°X — [titre]**
- Question posée : *(ex. le prétraitement change-t-il la performance d'un modèle simple ?)*
- Méthodes comparées :
- Protocole (jeu de données, split, métrique) :
- Résultat attendu / hypothèse :
- Résultat observé :
- Ce que ça démontre (force / faiblesse mise en évidence) :

*(à compléter après la réunion de mise à niveau)*

**Repère de planning (issu du cours, à adapter à votre rythme) :** veille + rédaction (1j) → première EDA (½j) → fonction `clean()` (½j) → deuxième EDA sur texte nettoyé (½j) → baseline TF-IDF + régression logistique (½j) → Decision Tree maison (1j) → GridSearch sur les 5 modèles (1j) → évaluation/conclusion (1j). Utile pour caler des scénarios de démonstration réalistes à date fixe plutôt que pour dicter le contenu des scénarios eux-mêmes.
