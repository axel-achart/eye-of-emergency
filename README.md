# Eye of Emergency

**Classification automatique de tweets signalant des catastrophes naturelles — Detection de fake news en periode de crise.**

---

## Table des matieres

1. [Contexte du projet](#contexte-du-projet)
2. [Presentation des donnees](#presentation-des-donnees)
3. [Analyse exploratoire des donnees](#analyse-exploratoire-des-donnees)
4. [Pipeline de pretraitement NLP](#pipeline-de-pretraitement-nlp)
5. [Algorithmes et modeles utilises](#algorithmes-et-modeles-utilises)
6. [Comparaison Bagging vs Boosting](#comparaison-bagging-vs-boosting)
7. [Optimisation par GridSearchCV](#optimisation-par-gridsearchcv)
8. [Evaluation et choix du modele final](#evaluation-et-choix-du-modele-final)
9. [Conclusion](#conclusion)
10. [Veille realisee — NLP & Text Mining](#veille-realisee--nlp--text-mining)
11. [Structure du projet](#structure-du-projet)
12. [Installation et execution](#installation-et-execution)

---

## Contexte du projet

### Problematique

Les plateformes de reseaux sociaux sont aujourd'hui devenues une partie integrante de nos vies, nous fournissant des informations en temps reel sur les evenements qui se produisent autour de nous. Toutefois, cet essor s'est accompagne d'une **augmentation de la desinformation et des « fake news »**. En cas de catastrophe ou de crise, ces fausses informations peuvent etre particulierement dangereuses, car elles peuvent entraver les efforts d'intervention d'urgence et provoquer la panique au sein de la population.

### Objectif

Developper un **modele d'apprentissage automatique** capable de classer des tweets signalant des catastrophes naturelles. L'objectif est de distinguer les **tweets informatifs** (qui decrivent une catastrophe reelle) de ceux qui ne le sont pas (usage metaphorique, humour, rumeurs, etc.), afin d'aider les intervenants d'urgence et le public a acceder a des informations precises et fiables en periode de crise.

### Cas d'usage concret

| Tweet | Classe | Explication |
|---|---|---|
| *« Massive earthquake just hit San Francisco! Buildings collapsing #earthquake »* | **Catastrophe (1)** | Decrit un evenement reel. |
| *« My heart is on fire after that workout »* | **Non-catastrophe (0)** | Usage metaphorique du mot « fire ». |
| *« I love watching disaster movies on Netflix »* | **Non-catastrophe (0)** | Mention non urgente. |
| *« Flood warning issued for Miami-Dade County — evacuate immediately »* | **Catastrophe (1)** | Alerte legitime. |

---

## Presentation des donnees

### Jeu de donnees : Disaster Tweets

Le jeu de donnees utilise est le celebre **Disaster Tweets** (source Kaggle). Il se compose de deux fichiers :

| Jeu de donnees | Fichier | Nombre de tweets | Colonnes |
|---|---|---|---|
| **Entrainement** | `train_tweets.csv` | 7 613 | `id`, `keyword`, `location`, `text`, `target` |
| **Test** | `test_tweets.csv` | 3 263 | `id`, `keyword`, `location`, `text` |

### Description des colonnes

| Colonne | Description | Type |
|---|---|---|
| `id` | Identifiant unique pour chaque tweet. | Entier |
| `text` | Le contenu textuel du tweet (140 a 280 caracteres). | Texte |
| `location` | Localisation d'ou a ete envoye le tweet (peut etre vide). | Texte |
| `keyword` | Un mot-cle associe au tweet (ex : *earthquake*, *fire*, *flood*). | Texte |
| `target` | **Variable cible** : `1` = tweet concernant une catastrophe reelle, `0` = sinon. | Binaire |

> **Note** : Le jeu de test (`test_tweets.csv`) ne contient pas la colonne `target`. Les predictions sont destinees a etre soumises sur Kaggle pour evaluation.

### Qualite des donnees — Valeurs manquantes

| Colonne | NaN (Train) | % (Train) | NaN (Test) | % (Test) |
|---|---|---|---|---|
| `keyword` | 61 | 0,8 % | 26 | 0,8 % |
| `location` | 2 533 | 33,3 % | 1 105 | 33,9 % |
| `text` | 0 | 0 % | 0 | 0 % |
| `target` | 0 | 0 % | N/A | N/A |

La colonne `location` presente un taux de valeurs manquantes eleve (~33%) ; dans ce projet, elle n'est pas utilisee pour la modelisation (seul le `text` et le `keyword` sont exploites).

### Distribution des classes

Les classes sont **legerement desequilibrees** :

- **Non-catastrophe (0)** : ~57 % des tweets
- **Catastrophe (1)** : ~43 % des tweets

Ce desequilibre modere ne necessite pas de techniques de reechantillonnage lourdes (SMOTE, undersampling), mais justifie l'utilisation du **F1-score** comme metrique principale plutot que l'accuracy pure.

---

## Analyse exploratoire des donnees

L'analyse exploratoire complete est disponible dans le notebook `data_analysis.ipynb`. En voici les principaux enseignements.

### 1. Longueur des tweets

| Statistique | Valeur |
|---|---|
| Longueur moyenne (caracteres) | ~101 caracteres |
| Longueur mediane (caracteres) | ~107 caracteres |
| Nombre moyen de mots | ~15 mots |
| Tweet le plus court | 7 caracteres |
| Tweet le plus long | 157 caracteres |

Les tweets de catastrophe sont **legerement plus longs** en moyenne que les tweets non-catastrophe (+5 caracteres), probablement car ils contiennent davantage d'informations factuelles.

### 2. Mots les plus frequents par classe

**Catastrophe (target=1)** : *fire*, *storm*, *flood*, *earthquake*, *disaster*, *killed*, *death*, *police*, *news*, *via*

**Non-catastrophe (target=0)** : *new*, *one*, *like*, *get*, *body*, *would*, *time*, *love*, *feel*, *know*

On observe une **separation semantique claire** entre les deux classes : les tweets catastrophe contiennent des termes factuels lies aux urgences, tandis que les tweets non-catastrophe utilisent un vocabulaire plus general et subjectif.

### 3. Hashtags et mentions

- **#earthquake**, **#news**, **#storm**, **#fire** sont les hashtags les plus frequents dans la classe catastrophe.
- Les hashtags type **#love**, **#job**, **#life** dominent dans la classe non-catastrophe.

### 4. Analyse des keywords

Les mots-cles les plus discriminants pour la classe catastrophe sont : *derailment*, *wrecked*, *debris*, *suicide*, *bombing*, *ruined*, *evacuated*, *devastation*.

A l'inverse, des keywords comme *body* (body bag vs body language), *army* (military vs fan army), *ruin* (destroyed vs ruined my day) sont fortement ambigus et constituent les cas les plus difficiles a classifier.

---

## Pipeline de pretraitement NLP

Un pipeline de nettoyage en **9 etapes** a ete implemente dans la fonction `clean_text()` :

| Etape | Operation | Justification |
|---|---|---|
| **1. Suppression des URLs** | Retrait de `http://`, `https://`, `www.` | Les URLs sont du bruit, elles ne portent pas de sens lexical. |
| **2. Suppression des mentions** | Retrait de `@username` | Les noms d'utilisateurs ne sont pas informatifs pour la classification de catastrophes. |
| **3. Conservation du texte des hashtags** | `#earthquake` -> `earthquake` | Le symbole `#` est inutile, mais le mot lui-meme est informatif. |
| **4. Suppression de la ponctuation** | Retrait de `.,!?;:()[]{}` etc. | La ponctuation n'apporte pas d'information discriminante dans un modele vectoriel classique. |
| **5. Suppression des nombres** | Retrait des tokens purement numeriques | Les nombres isoles (dates, chiffres) ne sont pas porteurs de sens. |
| **6. Mise en minuscules** | Uniformisation de la casse | Evite que « Catastrophe » et « catastrophe » soient deux features differentes. |
| **7. Tokenisation** | Decoupage du texte en mots individuels | Etape indispensable pour toute analyse lexicale. |
| **8. Suppression des stop-words** | Filtrage des mots grammaticaux (the, a, is, etc.) | Reduction du bruit et de la dimensionnalite. |
| **9. Lemmatisation** | Reduction a la forme canonique (*killed* -> *kill*, *running* -> *run*) | Regroupe les variantes morphologiques, reduit la dispersion du vocabulaire. |

### Impact du pretraitement

- **Taux de reduction du vocabulaire** : ~40-50 % de tokens en moins apres nettoyage.
- **Meilleure separation semantique** : les nuages de mots apres nettoyage montrent une distinction plus nette entre les deux classes.
- **Comparaison lemmatisation vs stemming** : la lemmatisation produit des resultats legerement superieurs au stemming (F1-score +0.5 a +1 point).

### Vectorisation TF-IDF

Les textes nettoyes sont transformes en vecteurs numeriques via **TF-IDF** (Term Frequency — Inverse Document Frequency) avec les parametres suivants :

- `max_features=5000` : vocabulaire limite aux 5 000 termes les plus frequents.
- `ngram_range=(1, 2)` : unigrammes + bigrammes pour capturer les expressions de deux mots.
- `min_df=2` : un terme doit apparaitre dans au moins 2 documents.
- `max_df=0.9` : les termes apparaissant dans plus de 90 % des documents sont ignores.
- `sublinear_tf=True` : attenuation logarithmique des frequences.

> **Pourquoi TF-IDF ?** Cette ponderation donne plus de poids aux mots rares dans le corpus mais frequents dans un document donne, ce qui correspond bien au besoin d'identifier des termes discriminants comme « earthquake » ou « evacuation ».

---

## Algorithmes et modeles utilises

Cinq modeles ont ete entraines et compares :

### 1. Regression Logistique
- **Type** : Modele lineaire supervise.
- **Fonctionnement** : Estime la probabilite d'appartenance a la classe « catastrophe » via une fonction sigmoide appliquee a une combinaison lineaire des features TF-IDF.
- **Avantages** : Rapide, interpretable, performant sur les donnees textuelles lineairement separables.
- **Regularisation** : L2 (Ridge), parametre `C=1.0`.

### 2. Decision Tree (Arbre de decision)
- **Type** : Modele non-lineaire base sur des regles de decision hierarchiques.
- **Fonctionnement** : Divisions successives de l'espace des features selon le critere de Gini.
- **Avantages** : Interpretable visuellement, capture les interactions entre features.
- **Limites** : Tendance au surapprentissage, variance elevee.
- **Implementation custom** : Un arbre de decision binaire de type **CART** a ete implemente *from scratch* en Python pur (classe `CustomDecisionTree`), avec critere de Gini, profondeur et `min_samples_split` configurables.

### 3. Random Forest (Bagging)
- **Type** : Ensemble de N arbres de decision entraines en parallele sur des echantillons bootstrap.
- **Fonctionnement** : Agregation par vote majoritaire des predictions de chaque arbre.
- **Hyperparametres** : `n_estimators=100`, `max_depth=15`, `min_samples_split=5`.

### 4. XGBoost (Boosting)
- **Type** : Ensemble sequentiel d'arbres de decision optimise par descente de gradient.
- **Fonctionnement** : Chaque nouvel arbre corrige les erreurs residuelles du modele precedent.
- **Hyperparametres** : `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`.

### 5. SVM (Support Vector Machine)
- **Type** : Modele a marge maximale.
- **Fonctionnement** : Recherche de l'hyperplan optimal separant les classes dans l'espace TF-IDF.
- **Noyau** : Lineaire (`kernel='linear'`, `C=1.0`).

---

## Comparaison Bagging vs Boosting

| Critere | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| **Ordre d'entrainement** | Parallele (independant) | Sequentiel (dependant) |
| **Objectif principal** | Reduire la **variance** | Reduire le **biais** |
| **Poids des exemples** | Identiques | Ajustes a chaque iteration (erreurs sur-ponderees) |
| **Poids des modeles** | Egaux (vote majoritaire) | Ponderes par la performance |
| **Risque d'overfitting** | Faible | Plus eleve |
| **Sensibilite au bruit** | Robuste | Sensible |
| **Parallelisation** | Oui (chaque arbre independant) | Non (entrainement sequentiel) |
| **Interpretabilite** | Faible | Faible |

**Dans ce projet**, XGBoost surpasse legerement Random Forest (+1 a 2 points de F1-score), mais au prix d'un temps d'entrainement plus long et d'une sensibilite accrue aux hyperparametres.

---

## Optimisation par GridSearchCV

Une optimisation par **GridSearchCV** (validation croisee 3-fold, scoring F1) a ete realisee pour chaque modele :

| Modele | Grille de recherche | Meilleurs parametres |
|---|---|---|
| **Regression Logistique** | `C in {0.1, 1.0, 10.0}`, `solver in {liblinear, lbfgs}` | `C=1.0`, `penalty='l2'` |
| **Random Forest** | `n_estimators in {50, 100, 200}`, `max_depth in {10, 20, None}`, `min_samples_split in {2, 5, 10}` | `n_estimators=200`, `max_depth=20` |
| **XGBoost** | `n_estimators in {50, 100, 200}`, `max_depth in {3, 6, 9}`, `learning_rate in {0.01, 0.1, 0.3}` | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |
| **SVM** | `C in {0.1, 1.0, 10.0}`, `kernel in {linear, rbf}`, `gamma in {scale, auto}` | `kernel='linear'`, `C=1.0` |

### Impact du pretraitement sur les performances

Trois variantes de pretraitement ont ete comparees avec une Regression Logistique :

| Variante | F1-score |
|---|---|
| **Lemmatisation + sans stop-words** | **0.79** |
| Lemmatisation + avec stop-words | 0.77 |
| Stemming + sans stop-words | 0.78 |

La combinaison **lemmatisation + suppression des stop-words** offre le meilleur compromis.

---

## Evaluation et choix du modele final

### Metriques utilisees

| Metrique | Definition | Utilite dans ce projet |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Proportion globale de predictions correctes. |
| **Precision** | TP / (TP + FP) | Parmi les tweets predits « catastrophe », combien le sont vraiment ? Evite les fausses alertes. |
| **Recall** | TP / (TP + FN) | Parmi les vraies catastrophes, combien sont detectees ? Crucial pour ne pas rater une alerte. |
| **F1-score** | 2 x (P x R) / (P + R) | **Moyenne harmonique precision/recall — metrique optimisee.** |
| **ROC-AUC** | Aire sous la courbe ROC | Capacite globale du modele a discriminer les deux classes. |

> **Pourquoi le F1-score ?** Les classes sont legerement desequilibrees (57/43). Le F1-score offre un compromis entre **ne pas rater une catastrophe** (recall) et **eviter les fausses alertes** (precision). Maximiser le F1 revient a trouver le meilleur point d'equilibre.

### Resultats comparatifs

| Modele | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Regression Logistique** | 0.81 | 0.80 | 0.75 | **0.77** | 0.87 |
| **Random Forest** | 0.80 | 0.78 | 0.74 | 0.76 | 0.85 |
| **XGBoost** | 0.80 | 0.79 | 0.74 | 0.76 | 0.86 |
| **SVM** | 0.81 | 0.80 | 0.73 | 0.76 | 0.86 |
| **Decision Tree (custom)** | 0.75 | 0.74 | 0.66 | 0.70 | 0.74 |

### Modele retenu : Regression Logistique avec TF-IDF

**Justification du choix :**

1. **Performance competitive** : F1-score de 0.77, comparable a XGBoost et Random Forest (ecart < 2 points), avec une simplicite bien superieure.
2. **Interpretabilite cruciale** : Dans un contexte de detection de fake news, il est essentiel de pouvoir **expliquer pourquoi** un tweet est classe comme catastrophe. Les coefficients de la regression logistique permettent d'identifier directement les mots les plus predictifs (ex : « earthquake » -> coefficient fortement positif, « love » -> coefficient negatif).
3. **Rapidite** : Entrainement en quelques secondes, prediction quasi instantanee — ideal pour un flux temps reel.
4. **Robustesse et simplicite** : Maintenance facile, risque de surapprentissage maitrise grace a la regularisation L2.
5. **Adequation donnees/modele** : La bonne separation lineaire des classes (confirmee par les performances proches du SVM lineaire) rend un modele lineaire parfaitement adapte.

---

## Conclusion

### Resume du travail accompli

Ce projet a permis de developper un **systeme complet de classification de tweets** pour la detection de catastrophes naturelles, en suivant une methodologie rigoureuse :

1. **Analyse exploratoire approfondie** : distribution des classes, longueur des tweets, analyse des mots-cles, hashtags, mentions, n-uplets.
2. **Pipeline de pretraitement NLP robuste** : nettoyage, tokenisation, lemmatisation, suppression des stop-words, vectorisation TF-IDF.
3. **Comparaison de 5 modeles de classification** : Regression Logistique, Decision Tree, Random Forest, XGBoost, SVM.
4. **Implementation from scratch d'un arbre de decision** : classe `CustomDecisionTree` en Python pur avec critere de Gini.
5. **Optimisation par GridSearchCV** : recherche exhaustive des meilleurs hyperparametres pour chaque modele.
6. **Choix argumente du modele final** : Regression Logistique pour son excellent rapport performance/interpretabilite/simplicite.

### Performances du modele final

| Metrique | Score | Interpretation |
|---|---|---|
| **Accuracy** | 0.81 | 81 % des tweets correctement classes. |
| **Precision** | 0.80 | Parmi les tweets predits « catastrophe », 80 % le sont reellement. |
| **Recall** | 0.75 | 75 % des vraies catastrophes sont detectees. |
| **F1-score** | 0.77 | Bon compromis precision/recall. |
| **ROC-AUC** | 0.87 | Excellente capacite a discriminer les classes. |

### Limites et axes d'amelioration

- **Faux negatifs (~25 %)** : un quart des catastrophes reelles ne sont pas detectees. Piste : ajustement du seuil de decision pour privilegier le recall.
- **TF-IDF ignore l'ordre des mots** : des representations plus riches (Word2Vec, GloVe, FastText) ou contextuelles (BERT, RoBERTa) pourraient ameliorer les performances.
- **Donnees multimodales non exploitees** : les images eventuellement associees aux tweets n'ont pas ete utilisees, de meme que les metadonnees (horodatage, interactions).
- **Detection de la negation** : une piste d'amelioration importante serait l'integration de la gestion de la negation dans le pipeline (*« no fire »* vs *« fire »*).

### Prochaines etapes

- [ ] Test d'embeddings contextuels pre-entraines (BERT-base, DistilBERT) fine-tunes pour la tache.
- [ ] Ajustement du seuil de decision pour reduire les faux negatifs.
- [ ] Mise en production via une API REST (FastAPI) pour une utilisation en temps reel.
- [ ] Prediction sur le jeu de test et soumission Kaggle.

---

## Veille realisee — NLP & Text Mining

Un important travail de veille a ete realise, consultable integralement dans le notebook `notebook.ipynb`. En voici la synthese.

### 1. Text Mining et Natural Language Processing : definitions et articulation

| | Text Mining | Natural Language Processing (NLP) |
|---|---|---|
| **Objectif** | Extraire de la connaissance actionnable a partir de grands volumes de texte. | Comprendre et generer le langage humain automatiquement. |
| **Approche** | Top-down : on part d'un besoin metier. | Bottom-up : on part des unites linguistiques fondamentales. |
| **Profondeur** | Traitements de surface (bag-of-words, TF-IDF, n-grammes). | Structure profonde : syntaxe, semantique, pragmatique. |
| **Granularite** | Niveau document ou corpus. | Tous niveaux : mot, phrase, discours. |
| **Resultat** | Indicateurs, clusters, tendances, classements. | Annotations linguistiques, traductions, reponses. |

**Formule d'articulation** :
> *Text Mining = NLP (pretraitement + extraction de caracteristiques) + Data Mining (modelisation + recherche de motifs)*

**Sources principales** :
- Feldman, R., & Sanger, J. (2007). *The Text Mining Handbook*. Cambridge University Press.
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Stanford.
- Weiss, S. M. et al. (2005). *Text Mining: Predictive Methods for Analyzing Unstructured Information*. Springer.

### 2. Principaux sous-domaines du NLP

| Sous-domaine | Description | Exemple |
|---|---|---|
| **Analyse de sentiments** | Determination automatique de la polarite emotionnelle d'un texte (positif/negatif/neutre). | « J'ai adore ce film ! » -> Positif |
| **Reconnaissance d'entites nominees (NER)** | Localisation et classification des entites (personnes, lieux, organisations, dates) dans un texte. | « Elon Musk a fonde SpaceX en 2002. » -> PER: Elon Musk, ORG: SpaceX, DATE: 2002 |
| **Etiquetage morpho-syntaxique (POS Tagging)** | Attribution de la categorie grammaticale a chaque mot. | « Le chat dort. » -> Le/DET, chat/NOUN, dort/VERB |
| **Tokenisation** | Segmentation du texte en unites elementaires (tokens). Peut etre au niveau mot, phrase ou sous-mot (BPE, WordPiece). | « Bonjour ! Ca va ? » -> ["Bonjour", "!", "Ca", "va", "?"] |

### 3. Stop-words : definition et enjeux

Un **stop-word** (ou mot vide) est un mot extremement frequent qui ne porte pas d'information semantique discriminante (ex : *le*, *la*, *de*, *the*, *a*, *is*).

| Impact | Description |
|---|---|
| **Reduction du bruit** | Les stop-words diluent le signal informatif. |
| **Reduction de la dimensionnalite** | Les stop-words representent 30 a 50 % des tokens d'un texte. |
| **Amelioration des modeles** | Leur suppression evite les correlations fallacieuses. |
| **Efficacite computationnelle** | Moins de features = entrainement plus rapide. |

> **Attention** : la suppression des stop-words n'est pas systematique. En analyse de sentiments, la negation (*« pas bon »*) est cruciale ; dans les Transformers (BERT), ils sont conserves pour la structure syntaxique.

**Source** : Manning, C. D., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

### 4. Ponctuation et caracteres speciaux

Le traitement de la ponctuation depend de la tache :
- **Modeles vectoriels classiques** (TF-IDF, Bag-of-Words) : suppression apres tokenisation.
- **Modeles Transformers** (BERT, GPT) : conservation en tant que tokens a part entiere.

Les caracteres speciaux (emojis, symboles monétaires, entites HTML, URLs) necessitent un traitement adapte :
- **Decodage HTML** (`&eacute;` -> `e`).
- **Normalisation Unicode** (NFC/NFD).
- **Reparation d'encodage** (ftfy pour le mojibake).
- **Gestion des emojis** : suppression, conversion en texte descriptif, ou conservation native.

**Source** : Unicode Consortium (2023). *The Unicode Standard, Version 15.0*.

### 5. Tokens et N-grammes

| Concept | Definition | Exemple (depuis « Le petit chat dort ») |
|---|---|---|
| **Token** | Unite elementaire issue de la segmentation. | ["Le", "petit", "chat", "dort"] |
| **Unigramme (1-gram)** | 1 token isole. | ["Le"], ["petit"], ["chat"], ["dort"] |
| **Bigramme (2-gram)** | 2 tokens consecutifs. | ["Le", "petit"], ["petit", "chat"], ["chat", "dort"] |
| **Trigramme (3-gram)** | 3 tokens consecutifs. | ["Le", "petit", "chat"], ["petit", "chat", "dort"] |

Les N-grammes capturent les **co-occurrences locales** et une partie de la structure syntaxique elementaire. Pour une sequence de M tokens, on genere M - N + 1 N-grammes par fenetre glissante.

**Source** : Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.), Chapitre 3.

---

## Structure du projet

```
eye-of-emergency/
|
├── data/
|   └── raw/
|       ├── train_tweets.csv          # Jeu d'entrainement (7 613 tweets etiquetes)
|       └── test_tweets.csv           # Jeu de test (3 263 tweets non etiquetes)
|
├── outputs/                          # Graphiques generes par l'analyse
|   ├── best_model_evaluation.png     # Matrice de confusion + courbe ROC du meilleur modele
|   ├── confusion_matrices.png        # Matrices de confusion des 5 modeles
|   ├── detailed_metrics_comparison.png # Comparaison detaillee des metriques
|   ├── keywords_by_class.png         # Top 15 keywords par classe
|   ├── missing_values.png            # Visualisation des valeurs manquantes
|   ├── preprocessing_impact.png      # Impact du pretraitement sur les performances
|   ├── target_distribution.png       # Distribution de la variable cible
|   ├── tweet_length_distribution.png # Distribution de la longueur des tweets
|   ├── wordcloud_analysis.png        # Nuages de mots avant nettoyage
|   └── wordcloud_cleaned.png         # Nuages de mots apres nettoyage
|
├── data_analysis.ipynb               # Notebook principal : analyse exploratoire et modelisation
├── notebook.ipynb                    # Notebook de veille : NLP, Text Mining, stop-words, etc.
├── README.md                         # Ce document
└── .gitignore
```

## Installation et execution

### Prerequis

- Python 3.8 ou superieur
- pip

### Installation des dependances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk wordcloud
```

### Execution

1. **Notebook d'analyse et de modelisation** :
   ```bash
   jupyter notebook data_analysis.ipynb
   ```

2. **Notebook de veille** :
   ```bash
   jupyter notebook notebook.ipynb
   ```

Les cellules s'executent sequentiellement. La premiere cellule du notebook d'analyse installe automatiquement les packages manquants.

---

## Auteurs

Projet realise dans le cadre du cursus **Bachelor IT — La Plateforme**, Marseille.

**Eye of Emergency** — Detection de fake news dans les tweets en periode de catastrophe.  
*Utiliser la data science au service de l'information fiable et de la securite civile.*

---

**Sources et references** : l'ensemble des sources academiques utilisees pour la veille est detaille dans le notebook `notebook.ipynb`.

---

---

# Eye of Emergency

**Automatic classification of tweets reporting natural disasters — Fake news detection in crisis situations.**

---

## Table of Contents

1. [Project Context](#project-context)
2. [Data Overview](#data-overview)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [NLP Preprocessing Pipeline](#nlp-preprocessing-pipeline)
5. [Algorithms and Models Used](#algorithms-and-models-used)
6. [Bagging vs Boosting Comparison](#bagging-vs-boosting-comparison)
7. [GridSearchCV Optimization](#gridsearchcv-optimization)
8. [Evaluation and Final Model Selection](#evaluation-and-final-model-selection)
9. [Conclusion](#conclusion-en)
10. [Literature Review — NLP & Text Mining](#literature-review--nlp--text-mining)
11. [Project Structure](#project-structure)
12. [Installation and Execution](#installation-and-execution)

---

## Project Context

### Problem Statement

Social media platforms have become an integral part of our lives, providing us with real-time information about events happening around us. However, this growth has been accompanied by an **increase in misinformation and "fake news"**. In the event of a disaster or crisis, this false information can be particularly dangerous, as it can hinder emergency response efforts and cause panic among the population.

### Objective

Develop a **machine learning model** capable of classifying tweets reporting natural disasters. The goal is to distinguish **informative tweets** (describing a real disaster) from non-informative ones (metaphorical usage, humor, rumors, etc.), in order to help emergency responders and the public access accurate and reliable information during crisis situations.

### Concrete Use Cases

| Tweet | Class | Explanation |
|---|---|---|
| *"Massive earthquake just hit San Francisco! Buildings collapsing #earthquake"* | **Disaster (1)** | Describes a real event. |
| *"My heart is on fire after that workout"* | **Non-disaster (0)** | Metaphorical use of the word "fire". |
| *"I love watching disaster movies on Netflix"* | **Non-disaster (0)** | Non-urgent mention. |
| *"Flood warning issued for Miami-Dade County — evacuate immediately"* | **Disaster (1)** | Legitimate alert. |

---

## Data Overview

### Dataset: Disaster Tweets

The dataset used is the well-known **Disaster Tweets** dataset (Kaggle source). It consists of two files:

| Dataset | File | Number of Tweets | Columns |
|---|---|---|---|
| **Training** | `train_tweets.csv` | 7,613 | `id`, `keyword`, `location`, `text`, `target` |
| **Test** | `test_tweets.csv` | 3,263 | `id`, `keyword`, `location`, `text` |

### Column Descriptions

| Column | Description | Type |
|---|---|---|
| `id` | Unique identifier for each tweet. | Integer |
| `text` | The textual content of the tweet (140 to 280 characters). | Text |
| `location` | Location from which the tweet was sent (may be empty). | Text |
| `keyword` | A keyword associated with the tweet (e.g. *earthquake*, *fire*, *flood*). | Text |
| `target` | **Target variable**: `1` = tweet about a real disaster, `0` = otherwise. | Binary |

> **Note**: The test set (`test_tweets.csv`) does not contain the `target` column. Predictions are intended for Kaggle submission and evaluation.

### Data Quality — Missing Values

| Column | NaN (Train) | % (Train) | NaN (Test) | % (Test) |
|---|---|---|---|---|
| `keyword` | 61 | 0.8% | 26 | 0.8% |
| `location` | 2,533 | 33.3% | 1,105 | 33.9% |
| `text` | 0 | 0% | 0 | 0% |
| `target` | 0 | 0% | N/A | N/A |

The `location` column has a high missing value rate (~33%); in this project, it is not used for modeling (only `text` and `keyword` are leveraged).

### Class Distribution

The classes are **slightly imbalanced**:

- **Non-disaster (0)**: ~57% of tweets
- **Disaster (1)**: ~43% of tweets

This moderate imbalance does not require heavy resampling techniques (SMOTE, undersampling), but justifies using the **F1-score** as the primary metric rather than pure accuracy.

---

## Exploratory Data Analysis

The complete exploratory analysis is available in the `data_analysis.ipynb` notebook. Here are the key findings.

### 1. Tweet Length

| Statistic | Value |
|---|---|
| Average length (characters) | ~101 characters |
| Median length (characters) | ~107 characters |
| Average word count | ~15 words |
| Shortest tweet | 7 characters |
| Longest tweet | 157 characters |

Disaster tweets are **slightly longer** on average than non-disaster tweets (+5 characters), likely because they contain more factual information.

### 2. Most Frequent Words by Class

**Disaster (target=1)**: *fire*, *storm*, *flood*, *earthquake*, *disaster*, *killed*, *death*, *police*, *news*, *via*

**Non-disaster (target=0)**: *new*, *one*, *like*, *get*, *body*, *would*, *time*, *love*, *feel*, *know*

A **clear semantic separation** is observed between the two classes: disaster tweets contain factual emergency-related terms, while non-disaster tweets use more general and subjective vocabulary.

### 3. Hashtags and Mentions

- **#earthquake**, **#news**, **#storm**, **#fire** are the most frequent hashtags in the disaster class.
- Hashtags like **#love**, **#job**, **#life** dominate the non-disaster class.

### 4. Keyword Analysis

The most discriminative keywords for the disaster class are: *derailment*, *wrecked*, *debris*, *suicide*, *bombing*, *ruined*, *evacuated*, *devastation*.

Conversely, keywords like *body* (body bag vs body language), *army* (military vs fan army), *ruin* (destroyed vs ruined my day) are highly ambiguous and constitute the most difficult cases to classify.

---

## NLP Preprocessing Pipeline

A **9-step** cleaning pipeline has been implemented in the `clean_text()` function:

| Step | Operation | Justification |
|---|---|---|
| **1. URL Removal** | Remove `http://`, `https://`, `www.` | URLs are noise; they carry no lexical meaning. |
| **2. Mention Removal** | Remove `@username` | Usernames are not informative for disaster classification. |
| **3. Hashtag Text Preservation** | `#earthquake` -> `earthquake` | The `#` symbol is useless, but the word itself is informative. |
| **4. Punctuation Removal** | Remove `.,!?;:()[]{}` etc. | Punctuation provides no discriminative information in a classical vector model. |
| **5. Number Removal** | Remove purely numeric tokens | Isolated numbers (dates, figures) are not meaningful. |
| **6. Lowercasing** | Case normalization | Prevents "Disaster" and "disaster" from being two different features. |
| **7. Tokenization** | Text segmentation into individual words | Essential step for any lexical analysis. |
| **8. Stop-word Removal** | Filter out grammatical words (the, a, is, etc.) | Reduces noise and dimensionality. |
| **9. Lemmatization** | Reduce to canonical form (*killed* -> *kill*, *running* -> *run*) | Groups morphological variants, reduces vocabulary dispersion. |

### Preprocessing Impact

- **Vocabulary reduction rate**: ~40-50% fewer tokens after cleaning.
- **Better semantic separation**: word clouds after cleaning show a sharper distinction between the two classes.
- **Lemmatization vs stemming comparison**: lemmatization produces slightly better results than stemming (F1-score +0.5 to +1 point).

### TF-IDF Vectorization

Cleaned texts are transformed into numerical vectors via **TF-IDF** (Term Frequency — Inverse Document Frequency) with the following parameters:

- `max_features=5000`: vocabulary limited to the 5,000 most frequent terms.
- `ngram_range=(1, 2)`: unigrams + bigrams to capture two-word expressions.
- `min_df=2`: a term must appear in at least 2 documents.
- `max_df=0.9`: terms appearing in more than 90% of documents are ignored.
- `sublinear_tf=True`: logarithmic frequency attenuation.

> **Why TF-IDF?** This weighting gives more importance to words that are rare in the corpus but frequent in a given document, which fits the need to identify discriminative terms like "earthquake" or "evacuation".

---

## Algorithms and Models Used

Five models were trained and compared:

### 1. Logistic Regression
- **Type**: Supervised linear model.
- **How it works**: Estimates the probability of belonging to the "disaster" class via a sigmoid function applied to a linear combination of TF-IDF features.
- **Advantages**: Fast, interpretable, performs well on linearly separable text data.
- **Regularization**: L2 (Ridge), parameter `C=1.0`.

### 2. Decision Tree
- **Type**: Non-linear model based on hierarchical decision rules.
- **How it works**: Successive splits of the feature space according to the Gini criterion.
- **Advantages**: Visually interpretable, captures feature interactions.
- **Limitations**: Tendency to overfit, high variance.
- **Custom implementation**: A binary **CART** decision tree was implemented *from scratch* in pure Python (`CustomDecisionTree` class), with configurable Gini criterion, depth, and `min_samples_split`.

### 3. Random Forest (Bagging)
- **Type**: Ensemble of N decision trees trained in parallel on bootstrap samples.
- **How it works**: Aggregation by majority vote of each tree's predictions.
- **Hyperparameters**: `n_estimators=100`, `max_depth=15`, `min_samples_split=5`.

### 4. XGBoost (Boosting)
- **Type**: Sequential ensemble of decision trees optimized by gradient descent.
- **How it works**: Each new tree corrects the residual errors of the previous model.
- **Hyperparameters**: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`.

### 5. SVM (Support Vector Machine)
- **Type**: Maximum margin model.
- **How it works**: Search for the optimal hyperplane separating classes in TF-IDF space.
- **Kernel**: Linear (`kernel='linear'`, `C=1.0`).

---

## Bagging vs Boosting Comparison

| Criterion | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| **Training Order** | Parallel (independent) | Sequential (dependent) |
| **Main Objective** | Reduce **variance** | Reduce **bias** |
| **Sample Weights** | Identical | Adjusted at each iteration (errors overweighted) |
| **Model Weights** | Equal (majority vote) | Weighted by performance |
| **Overfitting Risk** | Low | Higher |
| **Noise Sensitivity** | Robust | Sensitive |
| **Parallelization** | Yes (each tree independent) | No (sequential training) |
| **Interpretability** | Low | Low |

**In this project**, XGBoost slightly outperforms Random Forest (+1 to 2 F1-score points), but at the cost of longer training time and increased sensitivity to hyperparameters.

---

## GridSearchCV Optimization

**GridSearchCV** optimization (3-fold cross-validation, F1 scoring) was performed for each model:

| Model | Search Grid | Best Parameters |
|---|---|---|
| **Logistic Regression** | `C in {0.1, 1.0, 10.0}`, `solver in {liblinear, lbfgs}` | `C=1.0`, `penalty='l2'` |
| **Random Forest** | `n_estimators in {50, 100, 200}`, `max_depth in {10, 20, None}`, `min_samples_split in {2, 5, 10}` | `n_estimators=200`, `max_depth=20` |
| **XGBoost** | `n_estimators in {50, 100, 200}`, `max_depth in {3, 6, 9}`, `learning_rate in {0.01, 0.1, 0.3}` | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` |
| **SVM** | `C in {0.1, 1.0, 10.0}`, `kernel in {linear, rbf}`, `gamma in {scale, auto}` | `kernel='linear'`, `C=1.0` |

### Impact of Preprocessing on Performance

Three preprocessing variants were compared using Logistic Regression:

| Variant | F1-score |
|---|---|
| **Lemmatization + without stop-words** | **0.79** |
| Lemmatization + with stop-words | 0.77 |
| Stemming + without stop-words | 0.78 |

The **lemmatization + stop-word removal** combination offers the best trade-off.

---

## Evaluation and Final Model Selection

### Metrics Used

| Metric | Definition | Utility in this Project |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall proportion of correct predictions. |
| **Precision** | TP / (TP + FP) | Among tweets predicted as "disaster", how many actually are? Avoids false alarms. |
| **Recall** | TP / (TP + FN) | Among real disasters, how many are detected? Crucial so as not to miss an alert. |
| **F1-score** | 2 x (P x R) / (P + R) | **Harmonic mean of precision/recall — the optimized metric.** |
| **ROC-AUC** | Area under the ROC curve | Overall ability of the model to discriminate between the two classes. |

> **Why F1-score?** The classes are slightly imbalanced (57/43). The F1-score provides a trade-off between **not missing a disaster** (recall) and **avoiding false alarms** (precision). Maximizing F1 means finding the best balance point.

### Comparative Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.81 | 0.80 | 0.75 | **0.77** | 0.87 |
| **Random Forest** | 0.80 | 0.78 | 0.74 | 0.76 | 0.85 |
| **XGBoost** | 0.80 | 0.79 | 0.74 | 0.76 | 0.86 |
| **SVM** | 0.81 | 0.80 | 0.73 | 0.76 | 0.86 |
| **Decision Tree (custom)** | 0.75 | 0.74 | 0.66 | 0.70 | 0.74 |

### Selected Model: Logistic Regression with TF-IDF

**Rationale for choice:**

1. **Competitive performance**: F1-score of 0.77, comparable to XGBoost and Random Forest (gap < 2 points), with far greater simplicity.
2. **Crucial interpretability**: In a fake news detection context, it is essential to be able to **explain why** a tweet is classified as a disaster. Logistic regression coefficients directly identify the most predictive words (e.g. "earthquake" -> strongly positive coefficient, "love" -> negative coefficient).
3. **Speed**: Training in a few seconds, near-instant prediction — ideal for a real-time stream.
4. **Robustness and simplicity**: Easy maintenance, controlled overfitting risk thanks to L2 regularization.
5. **Data/model fit**: The good linear separation of classes (confirmed by the close performance of the linear SVM) makes a linear model perfectly suitable.

---

## Conclusion

### Summary of Completed Work

This project delivered a **complete tweet classification system** for natural disaster detection, following a rigorous methodology:

1. **In-depth exploratory analysis**: class distribution, tweet length, keyword/hashtag/mention/n-gram analysis.
2. **Robust NLP preprocessing pipeline**: cleaning, tokenization, lemmatization, stop-word removal, TF-IDF vectorization.
3. **Comparison of 5 classification models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM.
4. **From-scratch decision tree implementation**: `CustomDecisionTree` class in pure Python with Gini criterion.
5. **GridSearchCV optimization**: exhaustive search for the best hyperparameters for each model.
6. **Reasoned selection of the final model**: Logistic Regression for its excellent performance/interpretability/simplicity trade-off.

### Final Model Performance

| Metric | Score | Interpretation |
|---|---|---|
| **Accuracy** | 0.81 | 81% of tweets correctly classified. |
| **Precision** | 0.80 | Among tweets predicted as "disaster", 80% actually are. |
| **Recall** | 0.75 | 75% of real disasters are detected. |
| **F1-score** | 0.77 | Good precision/recall trade-off. |
| **ROC-AUC** | 0.87 | Excellent ability to discriminate between classes. |

### Limitations and Areas for Improvement

- **False negatives (~25%)**: a quarter of real disasters are not detected. Lead: adjust the decision threshold to favor recall.
- **TF-IDF ignores word order**: richer representations (Word2Vec, GloVe, FastText) or contextual embeddings (BERT, RoBERTa) could improve performance.
- **Unused multimodal data**: images potentially associated with tweets were not used, nor were metadata (timestamps, interactions).
- **Negation detection**: integrating negation handling into the pipeline would be an important improvement (*"no fire"* vs *"fire"*).

### Next Steps

- [ ] Test pre-trained contextual embeddings (BERT-base, DistilBERT) fine-tuned for the task.
- [ ] Adjust the decision threshold to reduce false negatives.
- [ ] Production deployment via a REST API (FastAPI) for real-time use.
- [ ] Prediction on the test set and Kaggle submission.

---

## Literature Review — NLP & Text Mining

Significant research work was carried out and is fully available in the `notebook.ipynb` notebook. Here is the synthesis.

### 1. Text Mining and Natural Language Processing: Definitions and Relationship

| | Text Mining | Natural Language Processing (NLP) |
|---|---|---|
| **Objective** | Extract actionable knowledge from large volumes of text. | Automatically understand and generate human language. |
| **Approach** | Top-down: starting from a business need. | Bottom-up: starting from fundamental linguistic units. |
| **Depth** | Surface processing (bag-of-words, TF-IDF, n-grams). | Deep structure: syntax, semantics, pragmatics. |
| **Granularity** | Document or corpus level. | All levels: word, sentence, discourse. |
| **Output** | Indicators, clusters, trends, rankings. | Linguistic annotations, translations, answers. |

**Relationship formula**:
> *Text Mining = NLP (preprocessing + feature extraction) + Data Mining (modeling + pattern discovery)*

**Key Sources**:
- Feldman, R., & Sanger, J. (2007). *The Text Mining Handbook*. Cambridge University Press.
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Stanford.
- Weiss, S. M. et al. (2005). *Text Mining: Predictive Methods for Analyzing Unstructured Information*. Springer.

### 2. Main NLP Subdomains

| Subdomain | Description | Example |
|---|---|---|
| **Sentiment Analysis** | Automatic determination of the emotional polarity of a text (positive/negative/neutral). | "I loved this movie!" -> Positive |
| **Named Entity Recognition (NER)** | Location and classification of entities (persons, places, organizations, dates) in a text. | "Elon Musk founded SpaceX in 2002." -> PER: Elon Musk, ORG: SpaceX, DATE: 2002 |
| **Part-of-Speech Tagging (POS Tagging)** | Assigning a grammatical category to each word. | "The cat sleeps." -> The/DET, cat/NOUN, sleeps/VERB |
| **Tokenization** | Text segmentation into elementary units (tokens). Can be at word, sentence, or subword level (BPE, WordPiece). | "Hello! How are you?" -> ["Hello", "!", "How", "are", "you", "?"] |

### 3. Stop-words: Definition and Challenges

A **stop-word** (or function word) is an extremely frequent word that carries little or no discriminative semantic information (e.g. *the*, *a*, *is*, *of*, *in*).

| Impact | Description |
|---|---|
| **Noise Reduction** | Stop-words dilute the informative signal. |
| **Dimensionality Reduction** | Stop-words account for 30 to 50% of text tokens. |
| **Model Improvement** | Their removal avoids spurious correlations. |
| **Computational Efficiency** | Fewer features = faster training. |

> **Warning**: stop-word removal is not systematic. In sentiment analysis, negation (*"not good"*) is crucial; in Transformers (BERT), they are preserved for syntactic structure.

**Source**: Manning, C. D., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

### 4. Punctuation and Special Characters

Punctuation handling depends on the task:
- **Classical vector models** (TF-IDF, Bag-of-Words): removal after tokenization.
- **Transformer models** (BERT, GPT): preserved as full tokens.

Special characters (emojis, currency symbols, HTML entities, URLs) require adapted processing:
- **HTML decoding** (`&eacute;` -> `e`).
- **Unicode normalization** (NFC/NFD).
- **Encoding repair** (ftfy for mojibake).
- **Emoji handling**: removal, conversion to descriptive text, or native preservation.

**Source**: Unicode Consortium (2023). *The Unicode Standard, Version 15.0*.

### 5. Tokens and N-grams

| Concept | Definition | Example (from "The little cat sleeps") |
|---|---|---|
| **Token** | Elementary unit from segmentation. | ["The", "little", "cat", "sleeps"] |
| **Unigram (1-gram)** | 1 isolated token. | ["The"], ["little"], ["cat"], ["sleeps"] |
| **Bigram (2-gram)** | 2 consecutive tokens. | ["The", "little"], ["little", "cat"], ["cat", "sleeps"] |
| **Trigram (3-gram)** | 3 consecutive tokens. | ["The", "little", "cat"], ["little", "cat", "sleeps"] |

N-grams capture **local co-occurrences** and part of the elementary syntactic structure. For a sequence of M tokens, M - N + 1 N-grams are generated via a sliding window.

**Source**: Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.), Chapter 3.

---

## Project Structure

```
eye-of-emergency/
|
├── data/
|   └── raw/
|       ├── train_tweets.csv          # Training set (7,613 labeled tweets)
|       └── test_tweets.csv           # Test set (3,263 unlabeled tweets)
|
├── outputs/                          # Graphs generated by the analysis
|   ├── best_model_evaluation.png     # Confusion matrix + ROC curve of the best model
|   ├── confusion_matrices.png        # Confusion matrices for the 5 models
|   ├── detailed_metrics_comparison.png # Detailed metrics comparison
|   ├── keywords_by_class.png         # Top 15 keywords by class
|   ├── missing_values.png            # Missing values visualization
|   ├── preprocessing_impact.png      # Preprocessing impact on performance
|   ├── target_distribution.png       # Target variable distribution
|   ├── tweet_length_distribution.png # Tweet length distribution
|   ├── wordcloud_analysis.png        # Word clouds before cleaning
|   └── wordcloud_cleaned.png         # Word clouds after cleaning
|
├── data_analysis.ipynb               # Main notebook: exploratory analysis and modeling
├── notebook.ipynb                    # Literature review notebook: NLP, Text Mining, stop-words, etc.
├── README.md                         # This document
└── .gitignore
```

## Installation and Execution

### Prerequisites

- Python 3.8 or higher
- pip

### Dependency Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk wordcloud
```

### Execution

1. **Analysis and modeling notebook**:
   ```bash
   jupyter notebook data_analysis.ipynb
   ```

2. **Literature review notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

Cells are executed sequentially. The first cell of the analysis notebook automatically installs any missing packages.

---

## Authors

Project carried out as part of the **Bachelor IT — La Plateforme** program, Marseille.

**Eye of Emergency** — Fake news detection in tweets during disaster situations.  
*Using data science to serve reliable information and civil security.*

---

**Sources and references**: all academic sources used for the literature review are detailed in the `notebook.ipynb` notebook.