# 🔍 Analyse Sémantique de Textes — Embeddings & Similarité

Ce module Python explore différentes **représentations vectorielles de textes** (embeddings) pour analyser la **proximité sémantique** entre des mots ou des documents à partir d’un corpus. Il combine des approches classiques (BoW, TF-IDF) et avancées (Word2Vec, BERT).

---

## 🎯 Objectifs

- Représenter les textes sous forme numérique (vectorielle).
- Étudier la fréquence et l’importance des mots avec **BoW** et **TF-IDF**.
- Apprendre des relations sémantiques avec **Word2Vec**.
- Extraire des représentations contextuelles puissantes avec **BERT**.
- Visualiser les similarités entre mots et documents via **cosine similarity**, **heatmaps**, et **t-SNE**.

---

## 📦 Contenu du Code

### 1. 🔡 BoW & TF-IDF

#### ➤ Pourquoi ?
- Analyser les mots présents dans chaque article.
- Mesurer leur importance relative.

#### ➤ Étapes :
- Charger les textes nettoyés (`cleaned_text`) depuis un DataFrame.
- Créer des représentations BoW (`CountVectorizer`) et TF-IDF (`TfidfVectorizer`).
- Afficher les matrices sous forme de `DataFrame`.
- Calculer manuellement le TF, IDF et TF-IDF d’un mot spécifique pour comparer avec `sklearn`.

---

### 2. 🧠 Word2Vec (Gensim)

#### ➤ Pourquoi ?
- Apprendre les relations sémantiques entre les mots (proximité, analogie, etc.).
- Découvrir les mots proches dans un espace vectoriel appris à partir du corpus.

#### ➤ Étapes :
- Tokeniser chaque article en liste de mots.
- Entraîner un modèle `Word2Vec` sur le corpus.
- Visualiser les similarités entre paires de mots.
- Extraire les mots les plus proches de certains mots-clés.
- Projeter les vecteurs appris dans un plan 2D avec `t-SNE`.

---

### 3. 🤖 BERT Multilingue (`bert-base-multilingual-cased`)

#### ➤ Pourquoi ?
- Capturer le **contexte** autour de chaque mot dans une phrase.
- Obtenir des embeddings de qualité supérieure, sensibles à la position et au sens du mot.

#### ➤ Étapes :
- Charger le modèle et le tokenizer BERT via `transformers`.
- Pour chaque mot cible, extraire son embedding dans chaque article.
- Calculer toutes les similarités cosinus entre paires de mots et d’articles.
- Organiser les résultats dans une `DataFrame` (pivotée ou plate).
- Moyenne des embeddings par mot pour visualisation globale.
- Visualiser les relations par **heatmap** et **t-SNE**.

---

## 📊 Visualisations

- **Matrice de similarité (heatmap)** : montre quels mots sont les plus proches en moyenne.
- **Projection t-SNE** : rend visible la structure sémantique dans un plan 2D.

---

## 🛠️ Comment utiliser ce script

### 1. Prérequis

Installez les bibliothèques nécessaires :

```bash
pip install pandas numpy sklearn gensim seaborn matplotlib transformers torch
```
  
### 2. 📂 Préparez votre corpus

Le script attend un DataFrame nommé `df_sample` contenant au minimum les colonnes suivantes :

- `cleaned_text` : le texte nettoyé de chaque document (minimisation, tokenisation, etc.)
- `section` : un identifiant, numéro ou titre associé au document (utile pour indexer les résultats)

**Exemple :**

```python
import pandas as pd

df_sample = pd.DataFrame({
   "section": ["Article 1", "Article 2"],
   "cleaned_text": [
      "iran signed trade agreement with china",
      "rugby world cup attracts millions of viewers"
   ]
})
```

### 3. 🎯 Modifiez les mots cibles (optionnel)

Vous pouvez personnaliser la liste des mots à analyser avec Word2Vec et BERT en modifiant la variable :

```python
target_words = ["rugby", "citizen", "arab", "emirate", "signed", "trade", "village", "songwriter", "thunder", "nba", "fertility", "maker", "report"]
```

Ces mots seront utilisés pour :

- Extraire leurs vecteurs d'embedding dans différents contextes
- Comparer leur similarité avec d'autres mots
- Visualiser leur position dans un espace réduit (t-SNE)

---

## 🧪 Résultats observables

Le script compare les scores de similarité sémantique produits par différentes méthodes :

- **BoW** : sensible à la fréquence brute (présence de mots)
- **TF-IDF** : pondère par rareté du mot (informativité)
- **Word2Vec** : apprend des relations sémantiques basées sur le contexte global
- **BERT** : capture la signification contextuelle exacte dans chaque phrase

---

## 📁 Exemple de sortie

### 🔥 Heatmap de similarité moyenne BERT

Affiche les similarités moyennes entre les mots cibles :

```
Cosine Similarity Between Words (Average Embeddings)
```

### 📏 Comparaison TF-IDF manuel vs sklearn

Affiche pour un mot donné les calculs de :

```
TF-IDF (manuel) = 0.212
TF-IDF (sklearn) = 0.213
```

### 🔗 Word2Vec

Mesure de similarité entre mots (apprise depuis le corpus) :

```
'arab' ↔ 'emirate' : 0.842
```

### 🧠 Visualisation t-SNE de BERT

Réduction de dimension 2D des embeddings BERT :

```
🧠 t-SNE – BERT Embeddings par mot et article
```

---

## 📌 À personnaliser

- **Corpus** : Ajoutez ou remplacez vos propres documents dans `df_sample`
- **Word2Vec** : Changez les paramètres `vector_size`, `window`, `min_count`, etc.
- **TF-IDF** : Ajoutez des `stop_words`, modifiez `ngram_range` ou `max_df`
- **BERT** : Testez d’autres modèles comme `bert-base-uncased`, `camembert-base`, etc.

---

## 📬 Contact

Pour toute question ou suggestion d'amélioration, n'hésitez pas à me contacter ou ouvrir une issue si ce projet est intégré dans un dépôt GitHub.

---

## 🤔 Pourquoi privilégier BERT pour la classification d’articles de presse ?

Pour un projet de **classification d’articles de presse selon leur contenu**, il est généralement plus pertinent d’utiliser **BERT** plutôt que des méthodes classiques comme BoW, TF-IDF ou même Word2Vec. 

**Pourquoi ?**

- **Compréhension contextuelle** : BERT prend en compte le contexte complet de chaque mot dans la phrase, ce qui permet de mieux saisir la signification réelle des textes journalistiques, souvent riches et nuancés.
- **Gestion des ambiguïtés** : Les articles de presse contiennent fréquemment des mots polysémiques (ayant plusieurs sens possibles) ou ambigus ; BERT adapte l’embedding de chaque mot à son contexte précis.
- **Performances supérieures** : Les modèles basés sur BERT atteignent généralement de meilleurs scores de classification sur des tâches de compréhension de texte, car ils capturent des relations sémantiques complexes.
- **Multilinguisme** : Avec des variantes comme `bert-base-multilingual-cased`, il est possible de traiter des articles dans plusieurs langues sans entraîner un modèle distinct pour chaque langue.

En résumé, **BERT** offre des représentations plus riches et adaptées à la complexité des articles de presse, ce qui améliore la précision de la classification par rapport aux approches traditionnelles.
