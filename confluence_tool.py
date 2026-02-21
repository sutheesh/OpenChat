"""
confluence_tool.py
The search_confluence(query) tool for Pattern 2 integration.
Plugs directly into your existing Flask app alongside the weather tool.
"""

import json
from confluence_rag import ConfluenceRAG

# Singleton - load once, reuse
_rag: ConfluenceRAG | None = None


def get_rag() -> ConfluenceRAG:
    global _rag
    if _rag is None:
        _rag = ConfluenceRAG()
    return _rag


def search_confluence(query: str, top_k: int = 5) -> dict:
    """
    Search the Confluence knowledge base for information.
    Returns relevant content with source URLs.
    """
    try:
        rag = get_rag()

        if rag.collection.count() == 0:
            return {
                "error": "Knowledge base is empty. Please run ingest.py first.",
                "results": []
            }

        results = rag.search(query, top_k=top_k)
        context = rag.format_context(results)

        return {
            "query": query,
            "results": results,
            "context": context,
            "total_chunks_searched": rag.collection.count()
        }

    except Exception as e:
        return {"error": str(e), "results": []}


# ── Tool definition for Pattern 2 (add to your tools list in app.py) ─────────
CONFLUENCE_TOOL_DEFINITION = {
    "name": "search_confluence",
    "description": "Search the internal knowledge base (Confluence) for information, documentation, guides, or any internal content. Use this when the user asks about documentation, processes, guides, or internal knowledge.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query - describe what information you're looking for"
            }
        },
        "required": ["query"]
    }
}


if __name__ == "__main__":
    # Quick test
    result = search_confluence("How to integrate Confluence with Slack?")
    print(json.dumps(result, indent=2))
