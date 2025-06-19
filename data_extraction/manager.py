import os
import pandas as pd
from guardian_api.fetcher import fetch_articles_for_section
from config.settings import SECTIONS, OUTPUT_CSV

def load_or_fetch_articles():
    if os.path.exists(OUTPUT_CSV):
        print(f"📄 Le fichier {OUTPUT_CSV} existe déjà.")
        return pd.read_csv(OUTPUT_CSV)
    
    print("📡 Extraction des articles depuis l'API The Guardian...")
    all_data = []
    for section in SECTIONS:
        print(f"→ Section: {section}")
        articles = fetch_articles_for_section(section)
        all_data.extend(articles)
    
    df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Données sauvegardées dans {OUTPUT_CSV}")
    return df
