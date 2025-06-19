# Prétraitement des textes pour la classification de news

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