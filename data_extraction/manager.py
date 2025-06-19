import os
import pandas as pd
from data_extraction.fetcher import fetch_articles_for_section
from config.settings import SECTIONS

def load_or_fetch_articles(output_csv):
    if os.path.exists(output_csv):
        print(f"📄 Le fichier {output_csv} existe déjà.")
        return pd.read_csv(output_csv)
    
    print("📡 Extraction des articles depuis l'API The Guardian...")
    all_data = []
    for section in SECTIONS:
        print(f"→ Section: {section}")
        articles = fetch_articles_for_section(section)
        all_data.extend(articles)
    
    df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Données sauvegardées dans {output_csv}")
    return df
