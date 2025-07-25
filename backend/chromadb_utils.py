import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client(Settings())
collection = chroma_client.get_or_create_collection("brand_content")

# Use a sentence transformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_content_to_chromadb(content_id: str, content_text: str, metadata: dict = None):
    embedding = embedder.encode([content_text])[0]
    collection.add(
        ids=[content_id],
        embeddings=[embedding],
        documents=[content_text],
        metadatas=[metadata or {}]
    )

def query_chromadb(query_text: str, top_k: int = 5):
    embedding = embedder.encode([query_text])[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return results 