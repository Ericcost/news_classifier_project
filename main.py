# Importation des modules nécessaires
import os
import pandas as pd
from config.settings import OUTPUT_CSV  # Chemin du fichier CSV de sortie défini dans la config
from nlp.preprocessing import Preprocessing  # Classe de prétraitement du texte
from data_extraction.manager import load_or_fetch_articles  # Fonction d'extraction des articles

def run_preprocessing():
    # Vérifie si le fichier CSV d'articles existe
    if not os.path.exists(OUTPUT_CSV):
        print(f"❌ Le fichier {OUTPUT_CSV} est introuvable. Lance d'abord l'extraction.")
        return
    
    # Charge les données depuis le CSV
    df = pd.read_csv(OUTPUT_CSV)
    preprocessor = Preprocessing()  # Initialise le préprocesseur
    # Applique le prétraitement à la colonne 'text'
    df['cleaned_text'] = df['text'].apply(preprocessor.preprocess)
    
    # Sauvegarde le résultat dans un nouveau fichier CSV
    output_cleaned = OUTPUT_CSV.replace(".csv", "_cleaned.csv")
    df.to_csv(output_cleaned, index=False)
    print(f"✅ Texte prétraité sauvegardé dans {output_cleaned}")

def menu():
    # Affiche le menu principal et récupère le choix de l'utilisateur
    print("Quelle étape veux-tu lancer ?")
    print("1 - Extraction des données (API Guardian)")
    print("2 - Prétraitement du texte")
    print("3 - Quitter")
    return input("Ton choix (1/2/3) : ").strip()

def main():
    # Fonction principale qui gère le menu et les actions
    choice = menu()
    
    if choice == "1":
        load_or_fetch_articles()  # Lance l'extraction des articles
    elif choice == "2":
        run_preprocessing()  # Lance le prétraitement du texte
    elif choice == "3":
        print("👋 Fin du script.")  # Quitte le programme
    else:
        print("❌ Choix invalide. Réessaie.")  # Gère les choix invalides

if __name__ == "__main__":
    main()  # Point d'entrée du script
