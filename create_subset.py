import pandas as pd
import os
from pathlib import Path

csv_path = Path(".") / "data" / "articles3.csv"
df = pd.read_csv(csv_path)

import sys
sys.path.append('../nlp') 

from nlp.preprocessing import Preprocessing

preprocessor = Preprocessing()

# S'assurer que la colonne texte n'a pas de valeurs manquantes
df = df[df['text'].notnull()].copy()

# Initialiser la colonne 'clean_text' vide
df['clean_text'] = None

# Appliquer le traitement sur les premiers articles de chaque section
for section in df['section'].unique():
    # Récupère les indices des premiers articles de la section
    section_indices = df[df['section'] == section].head(1).index
    df.loc[section_indices, 'clean_text'] = df.loc[section_indices, 'text'].apply(preprocessor.preprocess)

# Check how many were processed
print(df['clean_text'].notnull().sum())

from IPython.display import display

# Filtrer uniquement les lignes où le prétraitement a été appliqué
df_subset = df[df['clean_text'].notnull()][['section', 'title' ,'text', 'clean_text']]
pd.set_option('display.max_colwidth', 200)

display(df_subset)

# Save to CSV
output_path = os.path.join("data", "subset_articles3.csv")
df_subset.to_csv(output_path, index=False)

print(f"✅ Fichier sauvegardé dans : {output_path}")