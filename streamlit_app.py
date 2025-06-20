import sys
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="NLP News Analyzer", layout="centered")

#st.write("Python version used by Streamlit:", sys.version)
#st.write("PYTHONPATH:", sys.path)
#st.write("Current dir:", os.path.abspath(os.path.dirname(__file__)))

# Assure-toi que le chemin courant est dans sys.path pour importer ton module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from nlp.bert_analyzer import BERTSemanticAnalyzer
from nlp.preprocessing import Preprocessing
from vis.visualizer import Visualizer
from data_extraction.manager import load_or_fetch_articles

st.title("📰 Analyse d'articles de presse")

menu = st.sidebar.selectbox(
    "Menu",
    ["Accueil", "Extraire les données", "Prétraiter les données", "Tester le prétraitement", "Visualisation des articles", "Analyse sémantique BERT"]
)

st.write(f"## {menu}")

def ask_for_filename(default="data/articles.csv"):
    return st.text_input("Nom du fichier :", value=default)

if menu == "Accueil":
    st.write("Bienvenue sur l'application NLP News Analyzer ! Choisissez une action dans le menu.")

elif menu == "Extraire les données":
    st.subheader("Extraction via The Guardian API")
    filename = ask_for_filename("data/articles.csv")
    if st.button("Lancer l'extraction"):
        load_or_fetch_articles(filename)
        st.success(f"✅ Données sauvegardées dans {filename}")

elif menu == "Prétraiter les données":
    st.subheader("Prétraitement des données")
    filename = ask_for_filename("data/subset_articles3.csv")
    if st.button("Lancer le prétraitement"):
        if not os.path.exists(filename):
            st.error(f"❌ Le fichier {filename} est introuvable. Lance d'abord l'extraction.")
        else:
            df = pd.read_csv(filename)
            preprocessor = Preprocessing()
            df['cleaned_text'] = df['text'].apply(preprocessor.preprocess)
            output_cleaned = filename.replace(".csv", "_cleaned.csv")
            df.to_csv(output_cleaned, index=False)
            st.success(f"✅ Texte prétraité sauvegardé dans {output_cleaned}")

elif menu == "Tester le prétraitement":
    st.subheader("Test du prétraitement sur un exemple")
    # Exemple simple, tu peux adapter en fonction de ton test
    example_text = st.text_area("Texte à prétraiter :", "Breaking News! The U.S. economy grew by 3.2% in Q2 2023. Investors are excited — stocks soared!!! But beware of risks... Contact us at info@example.com or visit https://news.com.")
    if st.button("Lancer le test"):
        preprocessor = Preprocessing()
        cleaned = preprocessor.preprocess(example_text)
        #st.write("**Texte original :**", example_text)
        st.success(f"**Texte nettoyé :** {cleaned}")

elif menu == "Visualisation des articles":
    st.subheader("Visualisation des articles Avant/Après prétraitement")
    filename = ask_for_filename("data/subset_articles3_cleaned.csv")

    if not os.path.exists(filename):
        st.error(f"❌ Le fichier {filename} est introuvable. Veuillez prétraiter d'abord.")
    else:
        df = pd.read_csv(filename)

        if 'cleaned_text' not in df.columns:
            st.error("❌ La colonne 'cleaned_text' est absente. Veuillez prétraiter d'abord.")
        else:
            max_idx = len(df) - 1

            # Initialisation des variables dans session_state
            if 'article_idx' not in st.session_state:
                st.session_state.article_idx = 0
            if 'show_visualization' not in st.session_state:
                st.session_state.show_visualization = False

            # Input pour choisir l'article
            st.session_state.article_idx = st.number_input(
                "Choisir l'index de l'article à visualiser",
                min_value=0, max_value=max_idx,
                value=st.session_state.article_idx,
                step=1
            )

            # Bouton pour afficher la visualisation
            if st.button("Afficher les visualisations"):
                st.session_state.show_visualization = True  # Active l'affichage

            # Affichage conditionnel basé sur le flag
            if st.session_state.show_visualization:
                row = df.iloc[st.session_state.article_idx]
                st.write(f"### Article {st.session_state.article_idx} - Section : {row.get('section', 'N/A')} \n **Titre :** {row.get('title', 'N/A')}")

                visualizer = Visualizer()
                figs = visualizer.display_analysis(row['text'], row['cleaned_text'], title=f"Article {st.session_state.article_idx}")
                for fig in figs:
                    st.pyplot(fig)

elif menu == "Analyse sémantique BERT":
    st.subheader("Analyse sémantique avec BERT")

    filename = ask_for_filename("data/subset_articles3_cleaned.csv")
    if not os.path.exists(filename):
        st.error(f"❌ Le fichier {filename} est introuvable. Veuillez prétraiter d'abord.")
    else:
        df = pd.read_csv(filename)
        if 'cleaned_text' not in df.columns:
            st.error("❌ La colonne 'cleaned_text' est absente. Veuillez prétraiter d'abord.")
        else:
            target_words_input = st.text_input(
                "Mots cibles (séparés par des virgules)",
                value="rugby,report,maker,trade,two"
            )
            target_words = [w.strip() for w in target_words_input.split(",") if w.strip()]
            
            if st.button("Calculer embeddings et similarités"):
                with st.spinner("Chargement du modèle BERT et calcul des embeddings..."):
                    bert = BERTSemanticAnalyzer()
                    all_embeddings = bert.compute_embeddings(df["cleaned_text"].tolist(), target_words)
                    
                    st.success("✅ Embeddings calculés.")
                    st.info("Calcul des similarités entre mots et articles...")
                    df_sim = bert.compute_pairwise_similarities(all_embeddings)
                    st.dataframe(df_sim)

                    st.info("Visualisation des similarités moyennes (heatmap)")
                    fig_heatmap = bert.plot_average_similarity_heatmap(all_embeddings)
                    st.pyplot(fig_heatmap)

                    st.info("Visualisation t-SNE des embeddings")
                    fig_tsne = bert.plot_tsne(all_embeddings)
                    if fig_tsne:
                        st.pyplot(fig_tsne)
                    else:
                        st.warning("Pas assez de vecteurs pour afficher le t-SNE.")



