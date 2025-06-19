import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['u', 'us', 'q'])
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Nettoyage du texte : minuscules, suppression HTML, ponctuation, chiffres, espaces multiples.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        # Normalisation unicode (accents etc.)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
        # Supprimer les URLs (http, https, www)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
        # Supprimer les emails
        text = re.sub(r'\S+@\S+', '', text)
    
        # Supprimer ponctuation et chiffres, garder lettres et espaces uniquement
        # (on enlève aussi les acronymes avec points en une fois)
        text = re.sub(r'\b[a-z]\.', '', text)  # enlever lettres suivies d'un point (ex: u.)
        text = re.sub(r'[^a-z\s]', ' ', text)  # garder lettres et espaces uniquement
    
        # Enlever espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    def tokenize(self, text):
        """
        Tokenisation simple par split des mots.
        """
        return text.split()
    
    def remove_stopwords(self, tokens):
        """
        Suppression des stopwords.
        """
        filtered = [token for token in tokens if token not in self.stop_words]  
        return filtered
    
    def lemmatize(self, tokens):
        """
        Lemmatisation des tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Pipeline complet combinant toutes les étapes
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        no_stop = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(no_stop)
        return " ".join(lemmas)
