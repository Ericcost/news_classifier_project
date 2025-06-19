import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns

class Visualizer:
    def __init__(self):
        pass

    def check_cleaned_text_column(self, df):
        print(df.columns)
        return 'cleaned_text' in df.columns

    def plot_lengths(self, original_text, cleaned_text, title=""):
        original_len = len(original_text.split())
        cleaned_len = len(cleaned_text.split())
        lengths = [original_len, cleaned_len]
        labels = ['Avant', 'Après']

        plt.figure(figsize=(5, 4))
        plt.bar(labels, lengths, color=['skyblue', 'orange'])
        plt.ylabel("Nombre de mots")
        plt.title(f"Longueur du texte - {title}")
        plt.show()

    def show_wordclouds(self, original_text, cleaned_text):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        wordcloud_orig = WordCloud(width=400, height=400, background_color='white').generate(original_text)
        axs[0].imshow(wordcloud_orig, interpolation='bilinear')
        axs[0].axis('off')
        axs[0].set_title("Avant prétraitement")

        wordcloud_clean = WordCloud(width=400, height=400, background_color='white').generate(cleaned_text)
        axs[1].imshow(wordcloud_clean, interpolation='bilinear')
        axs[1].axis('off')
        axs[1].set_title("Après prétraitement")

        plt.tight_layout()
        plt.show()

    def show_common_words_heatmap(self, original_text, cleaned_text):
        original_freq = Counter(original_text.split()).most_common(20)
        cleaned_freq = Counter(cleaned_text.split()).most_common(20)

        common_words = set([w for w, _ in original_freq]) & set([w for w, _ in cleaned_freq])
        data = []
        for w in common_words:
            count_before = dict(original_freq).get(w, 0)
            count_after = dict(cleaned_freq).get(w, 0)
            data.append({'word': w, 'Avant': count_before, 'Après': count_after})
        
        if not data:
            print("⚠️ Aucun mot commun significatif pour heatmap.")
            return

        df_freq = pd.DataFrame(data).set_index('word')

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_freq, annot=True, cmap='YlGnBu')
        plt.title("Fréquences des mots communs")
        plt.show()

    def display_analysis(self, original_text, cleaned_text, title=""):
        self.plot_lengths(original_text, cleaned_text, title)
        self.show_wordclouds(original_text, cleaned_text)
        self.show_common_words_heatmap(original_text, cleaned_text)

    def analyze_file(self, filepath):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"❌ Le fichier {filepath} est introuvable.")
            return

        if not self.check_cleaned_text_column(df):
            print("❌ La colonne 'cleaned_text' est absente. Veuillez prétraiter d'abord.")
            return

        for i, row in df.iterrows():
            print(f"\n=== Article index {i} - Section: {row.get('section', 'N/A')} ===")
            self.display_analysis(row['text'], row['cleaned_text'], title=f"Article {i}")
