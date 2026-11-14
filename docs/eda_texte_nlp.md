# EDA texte — la "bible" du data analyst NLP

Ce fichier est un canevas de méthode, pas un rapport tout fait : il liste les étapes d'une EDA appliquée au texte, avec pour chacune des questions à se poser, un exercice à faire sur `data/raw/train_tweets.csv`, et les sources à consulter. À toi de mener les calculs (dans `notebook.ipynb` par exemple) et de noter tes observations dans les champs prévus.

**Colonnes du dataset :** `id`, `keyword`, `location`, `text`, `target` (1 = catastrophe réelle, 0 = non).

**Sources de référence :**
1. [Kaggle — NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started) (le dataset et la tâche)
2. [Stanford IR Book — Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
3. [spaCy](https://spacy.io) (tokenisation, POS, NER, pipelines)
4. [NLTK Book](https://www.nltk.org/book/) — notamment ch.1 (fréquences, corpus) et ch.3 (Processing Raw Text)
5. [regex101](https://regex101.com) (tester tes expressions régulières de nettoyage)
6. [`B2_Eye-of-Emergency_Ressources.md.pdf`](consignes/B2_Eye-of-Emergency_Ressources.md.pdf) (ajouté le 29/07/2026) — liste de ressources classée par question du sujet
7. [`B2_HowTo_Du-tweet-brut-au-vecteur.pdf`](consignes/B2_HowTo_Du-tweet-brut-au-vecteur.pdf) (ajouté le 29/07/2026) — atelier qui déroule un pipeline nettoyer → tokeniser → normaliser → vectoriser → vérifier sur un tweet réel

Une EDA texte suit la même logique qu'une EDA classique (structure → distributions → relations → qualité), mais chaque étape doit être adaptée parce que la donnée est du texte libre, pas des colonnes numériques.

**Comment ce canevas se raccorde au pipeline vu en cours :** le HowTo découpe le travail en Nettoyer → Tokeniser → Normaliser → Vectoriser → Vérifier. Les étapes 4 et 5 ci-dessous correspondent à Nettoyer/Tokeniser et Normaliser ; la Vectorisation (BoW/TF-IDF) est traitée comme notion de modélisation dans [`veille_technique_nlp.md`](veille_technique_nlp.md) (section 2) plutôt qu'ici ; et « Vérifier » correspond au principe même de ce fichier : refaire l'EDA après le nettoyage (cf. note à l'étape 5 du HowTo, reprise ci-dessous).

---

## Étape 0 — Cadrage

- Qu'est-ce qu'une "observation" ici ? Qu'est-ce qu'une "feature" avant tout traitement ?
- Quelle est la variable cible, et est-ce un problème équilibré ou déséquilibré ?
- Que signifient les colonnes `keyword` et `location` ? Sont-elles toujours renseignées ?

*Observations :* (à compléter)

---

## Étape 1 — Structure et qualité brute (l'équivalent de `.info()` / `.describe()`)

**Sources :** Kaggle (comprendre le schéma du dataset), NLTK book ch.1

Questions à traiter :
- Combien de lignes, combien de valeurs manquantes par colonne (`keyword`, `location` surtout) ?
- Y a-t-il des doublons de `text` ? Des `text` vides ou quasi vides ?
- Quelle est la répartition de la cible (`target`) ? Le jeu est-il déséquilibré ?
- Quelles langues sont présentes dans `text` (uniquement de l'anglais ?) ?

**Exercice :** calcule ces indicateurs sur `train_tweets.csv` et note ce qui te surprend.

**Repère de cours (à vérifier sur tes propres calculs, pas à recopier) :** la ressource ajoutée indique 7 613 tweets en train / 3 263 en test, et un déséquilibre d'environ 43 % de `target = 1`. Le test Kaggle n'a pas de colonne `target` : impossible d'y calculer un F1, il faut découper le train pour évaluer localement.

*Observations :* (à compléter)

---

## Étape 2 — Statistiques descriptives du texte

**Sources :** NLTK book ch.1 (longueurs, comptages), ch.3 (traitement du texte brut)

Pour chaque tweet, on peut dériver des variables numériques classiques d'EDA :
- Longueur en caractères, longueur en mots (tokens)
- Nombre de mots uniques / taille du vocabulaire par tweet
- Nombre de hashtags, de mentions (`@`), d'URLs, de majuscules, de ponctuation, d'emojis

Questions :
- Comment ces distributions diffèrent-elles selon `target` (0 vs 1) ? Un tweet de catastrophe est-il en moyenne plus long, plus court, avec plus de hashtags ?
- Y a-t-il des valeurs aberrantes (tweets anormalement longs/courts) ?

**Exercice :** calcule ces variables et compare leurs distributions (histogrammes, boxplots) par classe.

*Observations :* (à compléter)

---

## Étape 3 — Analyse lexicale (fréquences, vocabulaire)

**Sources :** NLTK book ch.1 (FreqDist, n-grams), spaCy (tokenisation linguistique)

Questions :
- Quels sont les mots (hors stopwords) les plus fréquents, globalement puis par classe ?
- Quels bigrammes/trigrammes reviennent souvent ? Sont-ils informatifs ou du bruit ?
- La tokenisation naïve (`.split()`) suffit-elle, ou la tokenisation spaCy/NLTK change-t-elle des choses importantes sur des tweets (ponctuation collée, contractions, hashtags, emojis) ?

**Exercice :** compare une tokenisation "maison" (regex simple) à une tokenisation spaCy ou NLTK sur 10-20 tweets. Note les différences.

*Observations :* (à compléter)

---

## Étape 4 — Bruit et spécificités Twitter (le terrain de regex101)

**Sources :** regex101, NLTK book ch.3

Questions :
- Quels motifs récurrents faut-il repérer avant tout nettoyage : URLs, `@mentions`, `#hashtags`, emojis, caractères HTML échappés (`&amp;`), abréviations, fautes de frappe ?
- Pour chaque motif, quelle regex le capture correctement sans "manger" du texte utile ? (à tester sur regex101 avec de vrais tweets du dataset)
- Faut-il supprimer ces éléments ou les transformer en information (ex. : extraire le hashtag comme feature séparée plutôt que le jeter) ?

**Exercice :** écris et teste sur regex101 une regex pour chacun des motifs ci-dessus, en utilisant des exemples réels tirés de `train_tweets.csv`. Documente les regex retenues ici.

**Exemple travaillé en cours (à adapter, pas à recopier tel quel — le tweet du cours n'est pas dans ton dataset) :**
```python
import re
def clean(t):
    t = t.lower()
    t = re.sub(r'http\S+|www\.\S+', ' ', t)   # URL : jamais répétée, gonfle le vocabulaire sans info
    t = re.sub(r'@\w+', ' ', t)                # mention : le pseudo est du bruit
    t = t.replace('#', ' ')                    # hashtag : on garde le mot, on jette le #
    t = re.sub(r'[^a-z\s]', ' ', t)             # tout ce qui n'est ni lettre ni espace
    return re.sub(r'\s+', ' ', t).strip()
```
Sur le tweet du cours (« OMG!!! The #forest fire near La Ronge is spreading FAST http://t.co/aBc123 @firedept HELP!! »), cette fonction fait passer 94 caractères à 55 (~30 % du vocabulaire en moins). Question à te poser sur *tes* tweets : est-ce que `clean()` mange des mots utiles sur des cas que tu n'as pas encore vus (abréviations, argot, caractères non-ASCII) ?

**Piège à vérifier sur ta stop-list :** `not`, `no`, `nor` sont dans la stop-list NLTK par défaut. Les enlever fait que « not a real fire » et « a real fire » produisent le même vecteur — deux classes potentiellement opposées, rendues indiscernables. Vérifie si ce cas apparaît dans ton dataset avant de trancher.

*Observations / regex retenues :* (à compléter)

---

## Étape 5 — Normalisation : stemming vs lemmatisation

**Sources :** Stanford IR Book — Stemming and lemmatization

Questions (reformule avec tes mots après lecture de la source) :
- Quelle est la différence de principe entre stemming et lemmatisation ?
- Le stemming (Porter par ex.) peut sur-regrouper des mots dont le sens diverge (l'IR book donne l'exemple de "operate/operating/operational" → "oper"). Est-ce un risque pour la détection de fake news sur tweets courts ?
- Pour ce projet (textes courts, bruités, vocabulaire informel), lemmatisation ou stemming — et pourquoi ?

**Exercice :** applique un stemmer (ex. Porter via NLTK) et un lemmatiseur (ex. spaCy) sur les mêmes 15-20 tweets. Compare les résultats mot à mot et identifie les cas où l'un des deux "casse" le sens.

**Exemples vus en cours (à titre de repère — refais le test sur tes propres tweets) :**

| Mot | Porter stemmer | Lemmatizer | Verdict |
|---|---|---|---|
| fires | fire | fire | identique |
| spreading | spread | spread | identique (si POS=verbe fourni) |
| studies | studi | study | racine invalide côté stemmer |
| university | univers | university | sens détruit côté stemmer |
| better | better | good | seul le lemme le voit |

Point de vigilance : le `WordNetLemmatizer` de NLTK suppose un nom par défaut — sans `pos='v'` explicite, `lemmatize('spreading')` renvoie `'spreading'` inchangé au lieu de `'spread'`. Sans POS tag, la lemmatisation ne sert presque à rien sur les verbes.

**Ce que le cours recommande pour trancher :** ne pas choisir au feeling — mettre stemming et lemmatisation dans le même `GridSearchCV`, comparer les F1-scores obtenus, et justifier le choix par le chiffre plutôt que par la lisibilité supposée. La lisibilité (ex. wordcloud présenté à un jury) reste un critère valable mais secondaire au F1.

*Observations :* (à compléter)

---

## Étape 6 — Relation texte ↔ cible

Questions :
- Existe-t-il des mots ou des `keyword` quasi exclusifs à une classe (`target`=1 ou 0) ?
- `location` apporte-t-elle un signal exploitable, ou est-elle trop bruitée/manquante pour être utile ?
- Le déséquilibre de classes observé à l'étape 1 a-t-il un impact sur la lecture de ces différences (attention aux effectifs faibles) ?

**Exercice :** pour un échantillon de `keyword` fréquents, calcule le taux de `target`=1 associé. Repère les `keyword` les plus polarisés.

*Observations :* (à compléter)

---

## Étape 7 — Synthèse

À rédiger une fois les étapes précédentes complétées :
- 3 constats principaux sur la qualité et la structure du texte
- Le choix de prétraitement que tu recommandes (nettoyage à garder/jeter, stemming ou lemmatisation) et pourquoi
- Les pistes de features à tester en priorité pour la suite du projet
- Ce qui reste incertain et qu'il faudra vérifier avec un scénario de démonstration (cf. [`veille_technique_nlp.md`](veille_technique_nlp.md), étape 3)

**Rappel important (issu du cours, étape « Vérifier » du HowTo) : l'EDA de ce fichier doit être menée deux fois.** Une première passe sur le texte brut (étapes 1 à 6 ci-dessus), une seconde après avoir écrit ta fonction `clean()` — pour vérifier que le nettoyage n'a pas effacé le signal qui distingue `target=0` de `target=1` (ex. : wordcloud par classe qui se ressemblent trop, top mots qui ont disparu). C'est explicitement demandé par le sujet et souvent oublié.

*Synthèse :* (à compléter)
