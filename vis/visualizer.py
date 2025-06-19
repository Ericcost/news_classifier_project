import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns

class Visualizer:
    def __init__(self):
        pass

    def check_cleaned_text_column(self, df):
        # Utile pour debug
        print(df.columns)
        return 'cleaned_text' in df.columns

    def plot_lengths(self, original_text, cleaned_text, title=""):
        original_len = len(original_text.split())
        cleaned_len = len(cleaned_text.split())
        lengths = [original_len, cleaned_len]
        labels = ['Avant', 'Après']

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(labels, lengths, color=['skyblue', 'orange'])
        ax.set_ylabel("Nombre de mots")
        ax.set_title(f"Longueur du texte - {title}")
        plt.close(fig)  # Important pour ne pas afficher hors Streamlit
        return fig

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
        plt.close(fig)
        return fig

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
            return None

        df_freq = pd.DataFrame(data).set_index('word')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_freq, annot=True, cmap='YlGnBu', ax=ax)
        ax.set_title("Fréquences des mots communs")
        plt.close(fig)
        return fig

    def display_analysis(self, original_text, cleaned_text, title=""):
        figs = []
        figs.append(self.plot_lengths(original_text, cleaned_text, title))
        figs.append(self.show_wordclouds(original_text, cleaned_text))
        heatmap_fig = self.show_common_words_heatmap(original_text, cleaned_text)
        if heatmap_fig is not None:
            figs.append(heatmap_fig)
        return figs

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
            figs = self.display_analysis(row['text'], row['cleaned_text'], title=f"Article {i}")
            for fig in figs:
                fig.show()  # Ou autre selon usage local
