# setup.py
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
import os

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB (persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.create_collection("mmk_site")

# -------- Scrape website --------
def scrape_site(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text

def chunk_text(text, max_chars=1000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# Website content
url = "https://15kay.github.io/mmk/"
raw_text = scrape_site(url)
chunks = chunk_text(raw_text)

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"web_chunk_{i}"]
    )

# -------- Personal profile --------
profile_text = """
Name: Kgaugelo Mmakola
Email: kg.mmakola@outlook.com
LinkedIn: https://www.linkedin.com/in/kgaugelo/
GitHub: https://github.com/15kay
Bio: Kgaugelo Mmakola is a passionate Data Scientist and Software Engineer from South Africa, currently pursuing an Advanced Diploma in ICT Application Development at Walter Sisulu University. He focuses on AI, predictive modeling, software development, and fintech projects.
"""

collection.add(
    documents=[profile_text],
    ids=["profile"]
)

print(f"Setup complete: {len(chunks)} website chunks + profile stored ✅")
