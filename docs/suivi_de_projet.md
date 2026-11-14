# Suivi de projet — Eye of Emergency

Outil de détection de fake news dans des publications Twitter.

Ce fichier centralise le suivi du projet : une entrée par session de travail, avec la date, les sujets principaux abordés, et les décisions prises.

---

## 28/07/2026

**Sujets abordés :**
- Mise en place de ce fichier de suivi de projet
- Cadrage de la démarche de veille technique NLP / text mining

**Discussion et décisions :**

Le projet en est à sa phase de cadrage. Les données brutes sont disponibles (`data/raw/train_tweets.csv`, `data/raw/test_tweets.csv`) mais aucune méthode de traitement n'a encore été choisie.

Avant de trancher sur une approche technique, l'équipe a décidé de mener une veille technique structurée sur les notions et méthodes de NLP / text mining, organisée en trois étapes :

1. **Recherche et cartographie individuelle** — chaque membre explore et documente les notions et méthodes de NLP / text mining pertinentes pour le projet (prétraitement, représentation du texte, classification, détection de fake news, etc.).
2. **Réunion de mise à niveau** — partage des recherches de chacun, mise en commun du vocabulaire et des connaissances, identification des zones d'ombre.
3. **Scénarios de démonstration** — construction de cas concrets sur les données du projet pour illustrer les forces et faiblesses de chaque méthode recensée, en vue de la restitution de la veille.

**Prochaines étapes :**
- Démarrer la cartographie (voir [`veille_technique_nlp.md`](veille_technique_nlp.md))
- Fixer une date pour la réunion de mise à niveau
- Une fois la cartographie avancée, définir les scénarios de démonstration

En complément, création de [`eda_texte_nlp.md`](eda_texte_nlp.md) : canevas d'EDA appliquée au texte (l'équivalent d'une EDA classique mais pour du texte libre), organisé en étapes (structure/qualité, stats descriptives du texte, analyse lexicale, bruit Twitter/regex, stemming vs lemmatisation, relation texte↔cible, synthèse), avec questions et exercices à mener sur `train_tweets.csv` plutôt que des réponses toutes faites, conformément à la démarche de veille du projet.

---

## 29/07/2026

**Sujets abordés :**
- Ajout de deux ressources de cours au repo : [`B2_Eye-of-Emergency_Ressources.md.pdf`](consignes/B2_Eye-of-Emergency_Ressources.md.pdf) (liste de ressources classée par question du sujet, avec repères 🔴/🟡/⚪) et [`B2_HowTo_Du-tweet-brut-au-vecteur.pdf`](consignes/B2_HowTo_Du-tweet-brut-au-vecteur.pdf) (support d'atelier 1h déroulant le pipeline nettoyer → tokeniser → normaliser → vectoriser → vérifier sur un tweet réel).
- Mise à jour de [`veille_technique_nlp.md`](veille_technique_nlp.md) et [`eda_texte_nlp.md`](eda_texte_nlp.md) pour intégrer ces sources.

**Discussion et décisions :**

Les sources consultées et des points de vigilance factuels (pièges connus : stop-words qui suppriment les négations, formule idf de sklearn, fuite de données si vectorisation avant split, POS tag requis pour la lemmatisation NLTK, etc.) ont été ajoutés dans les deux fichiers, section par section. Les champs de réflexion propres à l'exercice (avantages/limites/pertinence pour le projet dans [`veille_technique_nlp.md`](veille_technique_nlp.md), observations/synthèse dans [`eda_texte_nlp.md`](eda_texte_nlp.md)) restent volontairement `(à compléter)` : le cours donne le "quoi retenir", pas le raisonnement appliqué au projet, qui doit être fait par l'étudiant.

Deux zones identifiées comme non couvertes par les ressources actuelles : les embeddings (Word2Vec/GloVe/BERT) en section 2 de la veille, et toute la section 4 (deep learning) — à rechercher séparément si le projet en a besoin.

**Prochaines étapes :**
- Aucune donnée n'a encore été chargée/traitée dans `notebook.ipynb` (toujours vide) : la phase de cadrage/veille précède le code.
- Démarrer l'étape 1 de l'EDA (structure/qualité brute) sur `train_tweets.csv` pour vérifier les repères du cours (7 613 lignes, ~43 % de `target=1`).
- Compléter les champs de réflexion de [`veille_technique_nlp.md`](veille_technique_nlp.md) avant la réunion de mise à niveau.
