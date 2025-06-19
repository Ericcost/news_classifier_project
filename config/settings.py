# settings.py
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GUARDIAN_API_KEY")
BASE_URL = "https://content.guardianapis.com/search"
MAX_PAGES = int(os.getenv("MAX_PAGES", 200))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", 50))
SECTIONS = ["business", "politics", "culture", "sport", "technology", "world"]
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "data/articles.csv")
