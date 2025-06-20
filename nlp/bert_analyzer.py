# bert_analyzer.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class BERTSemanticAnalyzer:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_word_embedding(self, sentence, target_word):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

        # Trouver les indices des tokens correspondant au target_word
        target_word = target_word.lower()
        matched_indices = [i for i, token in enumerate(tokens) if target_word in token.lower()]

        if not matched_indices:
            return None
        
        # Moyenne des embeddings des sous-tokens correspondant
        selected_embeddings = embeddings[matched_indices, :]
        mean_embedding = selected_embeddings.mean(dim=0)
        return mean_embedding.detach().numpy()

    def compute_embeddings(self, corpus, target_words):
        all_embeddings = {}
        for word in target_words:
            embeddings = []
            for text in corpus:
                emb = self.get_word_embedding(text, word)
                embeddings.append(emb)
            all_embeddings[word] = embeddings
        return all_embeddings

    def compute_pairwise_similarities(self, all_embeddings):
        rows = []
        for word1, word2 in [(w1, w2) for w1 in all_embeddings for w2 in all_embeddings]:
            for i, emb1 in enumerate(all_embeddings[word1]):
                if emb1 is None:
                    continue
                for j, emb2 in enumerate(all_embeddings[word2]):
                    if emb2 is None:
                        continue
                    sim = cosine_similarity([emb1], [emb2])[0][0]
                    rows.append({
                        "Word 1": word1,
                        "Article Word1": i + 1,
                        "Word 2": word2,
                        "Article Word2": j + 1,
                        "Similarity": round(sim, 3)
                    })
        return pd.DataFrame(rows)

    def plot_average_similarity_heatmap(self, all_embeddings):
        avg_embeddings = {
            word: np.mean([e for e in embs if e is not None], axis=0)
            for word, embs in all_embeddings.items()
            if any(e is not None for e in embs)
        }
        words = list(avg_embeddings.keys())
        matrix = np.array([avg_embeddings[w] for w in words])
        sim_matrix = cosine_similarity(matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(sim_matrix, xticklabels=words, yticklabels=words, annot=True, cmap="viridis", ax=ax)
        ax.set_title("Cosine Similarity Between Words (Average Embeddings)")
        plt.tight_layout()
        plt.close(fig)  # Important pour éviter l'affichage direct hors Streamlit
        return fig

    def plot_tsne(self, all_embeddings):
        vectors = []
        labels = []

        for word, embs in all_embeddings.items():
            for i, emb in enumerate(embs):
                if emb is not None:
                    vectors.append(emb)
                    labels.append(f"{word} (Doc {i+1})")

        if len(vectors) < 2:
            print("❌ Pas assez de vecteurs pour appliquer t-SNE.")
            return None

        X = np.array(vectors)
        perplexity = min(5, len(X) - 1)
        X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c='red', s=100, alpha=0.7)

        for i, label in enumerate(labels):
            ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]), fontsize=10)

        ax.set_title("t-SNE – BERT Embeddings par mot et article")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True)
        plt.tight_layout()
        plt.close(fig)
        return fig
