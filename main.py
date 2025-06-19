import os
import pandas as pd
from nlp.preprocessing import Preprocessing
from vis.visualizer import Visualizer
from nlp.test_preprocessor import run_test
from data_extraction.manager import load_or_fetch_articles

DEFAULT_FILENAME = "data/articles.csv"

def ask_for_filename(prompt="Nom du fichier CSV (par défaut: data/articles.csv): "):
    filename = input(prompt).strip()
    return filename if filename else DEFAULT_FILENAME

def run_preprocessing(filename):
    # Force le chemin vers le dossier ./data
    base_name = os.path.basename(filename)  # récupère juste le nom du fichier
    full_path = os.path.join("data", base_name)

    if not os.path.exists(full_path):
        print(f"❌ Le fichier {full_path} est introuvable. Lance d'abord l'extraction.")
        return

    df = pd.read_csv(full_path)
    preprocessor = Preprocessing()
    df['cleaned_text'] = df['text'].apply(preprocessor.preprocess)

    output_cleaned = full_path.replace(".csv", "_cleaned.csv")
    df.to_csv(output_cleaned, index=False)
    print(f"✅ Texte prétraité sauvegardé dans {output_cleaned}")


def menu():
    print("\n=== Menu ===")
    print("1 - Extraire les données (API Guardian)")
    print("2 - Prétraiter un fichier CSV")
    print("3 - Tester le prétraitement sur un exemple")
    print("4 - Visualiser des articles")
    print("5 - Quitter")
    return input("Ton choix (1-5) : ").strip()

def main():
    choice = menu()

    if choice == "1":
        filename = ask_for_filename("Nom du fichier de sortie : ")
        load_or_fetch_articles(filename)

    elif choice == "2":
        filename = ask_for_filename("Nom du fichier à prétraiter : ")
        run_preprocessing(filename)

    elif choice == "3":
        run_test()

    elif choice == "4":
        filename = ask_for_filename("Nom du fichier nettoyé à visualiser : ")
        base_name = os.path.basename(filename)  # récupère juste le nom du fichier
        full_path = os.path.join("data", base_name)
        visualizer = Visualizer()
        visualizer.analyze_file(full_path)

    elif choice == "5":
        print("👋 Fin du programme.")

    else:
        print("❌ Choix invalide.")

if __name__ == "__main__":
    main()
