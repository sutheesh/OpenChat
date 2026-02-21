"""
ingest.py
Run this ONCE (or when content updates) to:
1. Crawl all pages under the base URL
2. Chunk + embed + store in ChromaDB

Usage:
    python ingest.py                          # use default base URL
    python ingest.py --url https://...        # custom base URL
    python ingest.py --reset                  # wipe and re-index
    python ingest.py --from-file results.json # skip crawl, re-index from saved file
"""

import argparse
import json
import os
from confluence_crawler import crawl, save_crawl_results
from confluence_rag import ConfluenceRAG

DEFAULT_BASE_URL = "https://www.atlassian.com/software/confluence/resources"
CRAWL_CACHE_FILE = "crawl_results.json"


def main():
    parser = argparse.ArgumentParser(description="Ingest Confluence content into RAG")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL to crawl")
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages to crawl")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    parser.add_argument("--reset", action="store_true", help="Reset and re-index from scratch")
    parser.add_argument("--from-file", help="Skip crawl, load from JSON file")
    args = parser.parse_args()

    # ── Step 1: Get pages ────────────────────────────────────────────────────
    if args.from_file:
        print(f"📂 Loading from file: {args.from_file}")
        with open(args.from_file, "r") as f:
            pages = json.load(f)
        print(f"   Loaded {len(pages)} pages")

    elif os.path.exists(CRAWL_CACHE_FILE) and not args.reset:
        print(f"📂 Found cached crawl results: {CRAWL_CACHE_FILE}")
        print("   Use --reset to re-crawl. Loading from cache...")
        with open(CRAWL_CACHE_FILE, "r") as f:
            pages = json.load(f)
        print(f"   Loaded {len(pages)} cached pages")

    else:
        print(f"🕷 Starting crawl: {args.url}")
        pages = crawl(args.url, max_pages=args.max_pages, delay=args.delay)
        save_crawl_results(pages, CRAWL_CACHE_FILE)

    if not pages:
        print("❌ No pages to index!")
        return

    # ── Step 2: Ingest into ChromaDB ─────────────────────────────────────────
    print(f"\n🔧 Ingesting {len(pages)} pages into ChromaDB...")
    rag = ConfluenceRAG()
    rag.ingest(pages, reset=args.reset)

    # ── Step 3: Quick test search ─────────────────────────────────────────────
    print("\n🔍 Running test search...")
    results = rag.search("How to use Confluence integrations", top_k=3)
    print(f"\nTop results for 'How to use Confluence integrations':")
    for r in results:
        print(f"  [{r['score']}] {r['title']}")
        print(f"  {r['url']}")
        print(f"  {r['text'][:150]}...")
        print()

    print("\n✅ Ingestion complete! You can now run app.py")
    print(f"   Stats: {rag.stats()}")


if __name__ == "__main__":
    main()
