# NLP News Analyzer

Cette application Streamlit permet d’extraire, prétraiter, visualiser et analyser sémantiquement des articles de presse grâce à des outils de NLP modernes (prétraitement, BERT, visualisation).

## Fonctionnalités principales

- **Accueil**  
  Présentation de l’application et navigation.

- **Extraire les données**  
  Extraction d’articles via l’API The Guardian et sauvegarde dans un fichier CSV.

- **Prétraiter les données**  
  Nettoyage et normalisation des textes (suppression des urls, emails, ponctuation, stopwords, lemmatisation, etc.) puis sauvegarde dans un nouveau fichier.

- **Tester le prétraitement**  
  Test interactif du pipeline de prétraitement sur un texte de votre choix.

- **Visualisation des articles**  
  Affichage comparatif d’un article avant/après prétraitement, avec visualisations graphiques.

- **Analyse sémantique BERT**  
  Extraction d’embeddings BERT pour des mots cibles, calcul de similarités cosinus, affichage de heatmaps et visualisation t-SNE des représentations sémantiques.

## Lancer l’application

1. **Installer les dépendances**  
   Assurez-vous d’avoir Python 3.8+ et installez les paquets nécessaires :

   ```bash
   pip install -r requirements.txt
   ```

   (ou, si le fichier n’existe pas, installez manuellement :  
   `pip install streamlit pandas transformers torch scikit-learn matplotlib seaborn`)

2. **Lancer l’application Streamlit**  
   Depuis le dossier du projet, exécutez :

   ```bash
   ## streamlit run streamlit_app.py
   python -m streamlit run streamlit_app.py
   ```

3. **Utiliser l’application**  
   Une fois l’application lancée, utilisez la barre latérale (menu à gauche) pour accéder aux différentes sections :  
   - **Accueil** : Présentation générale et instructions.  
     _(gérée dans `streamlit_app.py`)_
   - **Extraction des données** : Récupérez de nouveaux articles via l’API The Guardian.  
     _(fonctionnalités dans `data_extraction/manager.py`)_
   - **Prétraitement** : Nettoyez et normalisez les textes extraits.  
     _(pipeline dans `nlp/preprocessing.py`)_
   - **Test du prétraitement** : Essayez le pipeline sur un texte personnalisé.  
     _(utilise aussi `nlp/preprocessing.py`)_
   - **Visualisation** : Comparez les articles avant/après prétraitement et explorez les graphiques.  
     _(outils dans `vis/visualizer.py`)_
   - **Analyse sémantique BERT** : Analysez les similarités sémantiques et visualisez les embeddings.  
     _(fonctionnalités dans `nlp/bert_analyzer.py`)_

   Sélectionnez une section dans le menu pour afficher ses fonctionnalités et suivez les instructions à l’écran.

## Organisation du projet

- `streamlit_app.py` : Application principale Streamlit
- `nlp/preprocessing.py` : Pipeline de prétraitement des textes
- `nlp/bert_analyzer.py` : Analyse sémantique avec BERT
- `vis/visualizer.py` : Outils de visualisation
- `data_extraction/manager.py` : Extraction et gestion des articles

---

**Remarque :**  
Pour utiliser l’extraction d’articles, il peut être nécessaire de configurer une clé API The Guardian dans le code ou via une variable d’environnement.

---

**Auteur :**  
Eric Costerousse – YNOV M1 NLP Project

# Spécification concernant le Prétraitement des textes pour la classification d'articles de presse'

Ce module propose une classe [`Preprocessing`](nlp/preprocessing.py) qui applique un pipeline complet de nettoyage et de normalisation des textes. Ces étapes servent à préparer les données textuelles avant de les utiliser dans des modèles de NLP (Natural Language Processing).

## Étapes du prétraitement

1. **Conversion en minuscules**
   - **Pourquoi ?** : Uniformiser le texte pour éviter que le même mot soit traité différemment selon la casse (ex : "Apple" vs "apple").

2. **Normalisation Unicode**
   - **Pourquoi ?** : Supprimer les accents et caractères spéciaux pour ne garder que des caractères ASCII, ce qui simplifie l’analyse et évite les problèmes d’encodage.

3. **Suppression des URLs**
   - **Pourquoi ?** : Les liens web n’apportent généralement pas d’information utile pour la classification et peuvent introduire du bruit.

4. **Suppression des emails**
   - **Pourquoi ?** : Les adresses email sont rarement pertinentes pour l’analyse sémantique et peuvent aussi être considérées comme du bruit.

5. **Suppression de la ponctuation, des chiffres et des caractères spéciaux**
   - **Pourquoi ?** : On ne conserve que les lettres et les espaces pour se concentrer sur le contenu textuel pertinent.

6. **Suppression des acronymes d’une lettre suivis d’un point**
   - **Pourquoi ?** : Nettoyer les abréviations ou artefacts typographiques qui n’apportent pas de valeur pour la classification.

7. **Suppression des espaces multiples**
   - **Pourquoi ?** : Uniformiser les séparateurs de mots pour faciliter la tokenisation.

8. **Tokenisation**
   - **Pourquoi ?** : Découper le texte en mots (tokens) pour pouvoir appliquer les étapes suivantes sur chaque mot.

9. **Suppression des stopwords**
   - **Pourquoi ?** : Retirer les mots très fréquents (comme "the", "and", "is", etc.) qui n’apportent généralement pas d’information discriminante pour la classification.

10. **Lemmatisation**
    - **Pourquoi ?** : Réduire chaque mot à sa forme de base (ex : "running" → "run") pour regrouper les variantes d’un même mot et améliorer la généralisation du modèle.

## Exemple d’utilisation

```python
from nlp.preprocessing import Preprocessing

preprocessor = Preprocessing()
texte_propre = preprocessor.preprocess("Your raw text here.")
print(texte_propre)
```

## Fichier concerné

- [`nlp/preprocessing.py`](nlp/preprocessing.py)


# Spécification pour la partie Embeddings avec la classe BERTSemanticAnalyzer

Ce module propose la classe `BERTSemanticAnalyzer` pour l’analyse sémantique de mots dans des textes à l’aide de modèles BERT. Il permet d’extraire des embeddings de mots en contexte, de calculer des similarités cosinus, et de visualiser les relations sémantiques via des heatmaps et t-SNE.

## Étapes de traitement

### 1. **Tokenisation et vectorisation avec BERT**
- **Pourquoi ?**  
  La tokenisation BERT découpe le texte en sous-mots (subwords) pour gérer les mots inconnus et garantir une couverture linguistique maximale. Chaque token est ensuite converti en vecteur dense (embedding) par le modèle BERT, capturant le sens du mot dans son contexte.

### 2. **Extraction des embeddings du mot cible**
- **Pourquoi ?**  
  Pour chaque phrase et mot cible, on récupère les embeddings des sous-tokens correspondant au mot, puis on fait la moyenne pour obtenir une représentation unique du mot dans ce contexte. Cela permet d’analyser la polysémie et la variation de sens selon le contexte.

### 3. **Calcul des similarités cosinus**
- **Pourquoi ?**  
  La similarité cosinus mesure la proximité sémantique entre deux embeddings. Cela permet de comparer le sens d’un mot à travers différents textes ou de comparer différents mots entre eux.

### 4. **Visualisation par heatmap**
- **Pourquoi ?**  
  Une heatmap des similarités moyennes entre mots permet de visualiser rapidement les proximités et différences sémantiques globales dans le corpus.

### 5. **Réduction de dimension et t-SNE**
- **Pourquoi ?**  
  t-SNE réduit la dimensionnalité des embeddings pour permettre une visualisation 2D. Cela aide à explorer la structure sémantique des mots et à détecter des regroupements ou des anomalies.

## Exemple d’utilisation

```python
from nlp.bert_analyzer import BERTSemanticAnalyzer

analyzer = BERTSemanticAnalyzer()
corpus = ["Le chat dort.", "Le chien aboie."]
target_words = ["chat", "chien"]
embeddings = analyzer.compute_embeddings(corpus, target_words)
df_sim = analyzer.compute_pairwise_similarities(embeddings)
fig_heatmap = analyzer.plot_average_similarity_heatmap(embeddings)
fig_tsne = analyzer.plot_tsne(embeddings)
```

## Fichier concerné

- [`nlp/bert_analyzer.py`](nlp/bert_analyzer.py)