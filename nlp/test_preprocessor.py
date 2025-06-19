from nlp.preprocessing import Preprocessing
from vis.visualizer import Visualizer

def run_test():
    preprocessor = Preprocessing()
    visualizer = Visualizer()

    test_article = """
    Breaking News! The U.S. economy grew by 3.2% in Q2 2023.
    Investors are excited — stocks soared!!! But beware of risks...
    Contact us at info@example.com or visit https://news.com.
    """

    print("\nTexte original :\n")
    print(test_article)

    cleaned = preprocessor.preprocess(test_article)

    print("\nTexte après prétraitement :\n")
    print(cleaned)

    # Afficher les visualisations sur cet exemple
    visualizer.display_analysis(test_article, cleaned, title="Exemple Test")