"""
confluence_rag.py
Chunks crawled pages, embeds them, stores in ChromaDB.
Provides semantic search over all indexed content.
"""

import json
import re
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional


# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500        # characters (approx 125 tokens)
CHUNK_OVERLAP = 100     # characters overlap between chunks
COLLECTION_NAME = "confluence_kb"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, 80MB, good quality


# ── Chunker ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:   # skip tiny chunks
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ── RAG Engine ───────────────────────────────────────────────────────────────
class ConfluenceRAG:
    def __init__(self, db_path: str = DB_PATH):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Connecting to ChromaDB at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ RAG engine ready. Collection has {self.collection.count()} chunks.")

    def ingest(self, crawl_results: list[dict], reset: bool = False):
        """
        Ingest crawled pages into ChromaDB.
        crawl_results: list of {"url": str, "title": str, "text": str}
        """
        if reset:
            print("🗑 Resetting collection...")
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

        total_chunks = 0
        for page in crawl_results:
            url = page["url"]
            title = page["title"]
            text = page["text"]

            chunks = chunk_text(text)
            if not chunks:
                continue

            ids = [f"{url}__chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"url": url, "title": title, "chunk_index": i}
                         for i in range(len(chunks))]

            # Embed
            embeddings = self.embedder.encode(chunks).tolist()

            # Store (upsert to avoid duplicates on re-run)
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_chunks += len(chunks)
            print(f"  ✓ '{title}' → {len(chunks)} chunks")

        print(f"\n✅ Ingestion complete! {total_chunks} chunks stored.")
        print(f"   Total in DB: {self.collection.count()}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over indexed content.
        Returns list of dicts with: text, url, title, score
        """
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "url": results["metadatas"][0][i]["url"],
                "title": results["metadatas"][0][i]["title"],
                "score": round(1 - results["distances"][0][i], 3)  # cosine similarity
            })

        return output

    def format_context(self, search_results: list[dict]) -> str:
        """Format search results into a context string for the model."""
        if not search_results:
            return "No relevant content found."

        lines = []
        seen_urls = set()
        for r in search_results:
            lines.append(f"---")
            lines.append(f"Source: {r['title']}")
            lines.append(f"URL: {r['url']}")
            lines.append(f"Content: {r['text']}")
            seen_urls.add(r['url'])

        lines.append("---")
        lines.append(f"\nSources referenced:")
        for r in search_results:
            if r['url'] in seen_urls:
                lines.append(f"- {r['title']}: {r['url']}")
                seen_urls.discard(r['url'])

        return "\n".join(lines)

    def stats(self) -> dict:
        """Return stats about the knowledge base."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection": COLLECTION_NAME,
            "db_path": DB_PATH,
            "embedding_model": EMBEDDING_MODEL
        }


if __name__ == "__main__":
    # Test search
    rag = ConfluenceRAG()
    print("\nStats:", rag.stats())

    query = "How do I integrate Confluence with Slack?"
    print(f"\nSearching: '{query}'")
    results = rag.search(query, top_k=3)
    for r in results:
        print(f"\n[Score: {r['score']}] {r['title']}")
        print(f"URL: {r['url']}")
        print(f"Text: {r['text'][:200]}...")
