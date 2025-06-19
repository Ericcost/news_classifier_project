import requests
import time
from config.settings import API_KEY, BASE_URL, PAGE_SIZE, MAX_PAGES

def fetch_articles_for_section(section, max_pages=MAX_PAGES):
    all_articles = []
    
    for page in range(1, max_pages + 1):
        params = {
            "api-key": API_KEY,
            "section": section,
            "page-size": PAGE_SIZE,
            "page": page,
            "show-fields": "bodyText,headline"
        }
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code != 200:
            print(f"Erreur avec la section {section}, page {page}")
            break
        
        data = response.json()["response"]
        results = data.get("results", [])
        
        for article in results:
            all_articles.append({
                "section": article["sectionName"],
                "title": article["webTitle"],
                "text": article.get("fields", {}).get("bodyText", ""),
                #"link": article["webUrl"],
                #"date": article["webPublicationDate"]
            })
        
        time.sleep(0.2)  # pour éviter d'être bloqué par l'API

    return all_articles
