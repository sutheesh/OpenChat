# 🤖 AI Chat Assistant with Weather + Confluence RAG

A local AI chat assistant built with **Qwen 2.5-1.5B**, featuring real-time weather via **Apify MCP** and semantic search over any website/Confluence using **RAG (ChromaDB)**.

---

## ✨ Features

- 💬 **AI Chat** — Qwen 2.5-1.5B running fully on CPU (Mac Intel)
- 🌤️ **Weather Tool** — Live weather via Apify MCP (Model Context Protocol)
- 📚 **Confluence RAG** — Semantic search over any website or Confluence instance
- 📡 **Streaming** — Token-by-token streaming responses
- 🔧 **Pattern 2 Tool Calling** — Model decides when to use tools (no hardcoded rules)

---

## 🗂️ Project Structure

```
├── app.py                      # Main Flask app (entry point)
├── apify_weather_mcp.py        # Apify Weather MCP client
├── confluence_crawler.py       # Recursive web crawler
├── confluence_rag.py           # Chunking, embedding, ChromaDB search
├── confluence_tool.py          # search_confluence() tool definition
├── ingest.py                   # One-time crawl + index script
├── templates/
│   └── index.html              # Chat UI
├── static/                     # CSS, JS assets
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-chat-assistant.git
cd ai-chat-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your APIFY_TOKEN
```

### 5. Build the knowledge base (one time)
```bash
# Crawl and index the default URL
python ingest.py

# Or use your own URL
python ingest.py --url https://your-confluence.atlassian.net/wiki/spaces/SPACE

# Limit pages for a quick test
python ingest.py --max-pages 20
```

### 6. Run the app
```bash
python app.py
```

Open **http://localhost:8000** in your browser.

---

## ⚙️ Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
APIFY_TOKEN=<get your token>
```

Get your Apify token at: https://console.apify.com/account/integrations

---

## 🔧 Configuration

All settings are in `app.py` under `CONFIG`:

| Key | Default | Description |
|-----|---------|-------------|
| `base_model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model |
| `max_length` | `200` | Max tokens in response |
| `temperature` | `0.7` | Response creativity |
| `tool_temperature` | `0.0` | Tool decision (deterministic) |
| `max_tool_rounds` | `3` | Max tool call iterations |

RAG settings are in `confluence_rag.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

---

## 📚 Knowledge Base Management

```bash
# Initial crawl + index
python ingest.py --url https://your-site.com/docs

# Re-index without re-crawling (uses cached crawl_results.json)
python ingest.py --from-file crawl_results.json

# Wipe and re-index from scratch
python ingest.py --reset

# Check how many chunks are indexed
python -c "from confluence_rag import ConfluenceRAG; r = ConfluenceRAG(); print(r.stats())"
```

### Refresh via API endpoint
```bash
curl -X POST http://localhost:8000/api/refresh-kb
```

---

## 🔐 Using a Private Confluence Instance

If you have a real Atlassian Confluence Cloud instance, update `confluence_crawler.py`:

```python
import base64

EMAIL = "your@email.com"
API_TOKEN = "your_confluence_api_token"
credentials = base64.b64encode(f"{EMAIL}:{API_TOKEN}".encode()).decode()

headers = {
    "Authorization": f"Basic {credentials}",
    "User-Agent": "RAG-Crawler/1.0"
}
```

Get your Confluence API token: https://id.atlassian.com/manage-profile/security/api-tokens

---

## 🏗️ Architecture

```
User Query
    ↓
Flask Backend (app.py)
    ↓
Qwen 2.5-1.5B decides:
    ├── Weather query?   → Apify MCP → Open-Meteo API
    ├── Docs/knowledge?  → ChromaDB semantic search → Top 5 chunks
    └── Direct answer?   → Stream response
    ↓
Final answer streamed token by token
```

### Tool Calling Flow (Pattern 2)
```
1. User message
2. Model outputs JSON tool call  →  {"tool_call": {"name": "...", "arguments": {...}}}
3. Stream acknowledgment to user ("Let me search for that...")
4. Execute tool (weather API or ChromaDB)
5. Inject result into context
6. Stream final answer with sources
```

---

## 📊 Performance

| Component | Spec |
|-----------|------|
| Model | Qwen 2.5-1.5B (~3GB RAM) |
| Device | CPU (Mac Intel) |
| Inference speed | 2–5 tokens/sec |
| Weather tool latency | ~2–3 seconds |
| RAG search latency | <100ms |
| Total response time | 10–30 seconds |

---

## 🧪 API Reference

### `POST /api/chat`
```json
{
  "message": "How do I integrate Confluence with Slack?",
  "stream": true
}
```

**Streaming response** (`stream: true`): `text/event-stream`
```
data: {"token": "Let"}
data: {"token": " me"}
data: {"token": " search..."}
data: {"done": true}
```

**Non-streaming response** (`stream: false`):
```json
{
  "response": "To integrate Confluence with Slack...",
  "model": "Qwen 2.5-1.5B"
}
```

### `GET /api/status`
```json
{
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "CPU (Mac Intel)",
  "streaming_supported": true,
  "confluence_chunks": 1433
}
```

### `POST /api/refresh-kb`
Triggers a fresh crawl and re-index of the knowledge base.

---

## 🔜 Future Enhancements

- [ ] Multi-turn conversation memory
- [ ] More MCP servers (news, calendar, stocks)
- [ ] Cloud deployment (Render, Fly.io)
- [ ] Better UI with chat history
- [ ] Support for PDF / file uploads
- [ ] Scheduled KB refresh (cron)

---

## 📦 Dependencies

- `flask` — Web framework
- `torch` + `transformers` — Qwen model inference
- `sentence-transformers` — Text embeddings
- `chromadb` — Local vector database
- `httpx` + `beautifulsoup4` — Web crawling
- `peft` — Model adapter support

---

## 📄 License

MIT License — feel free to use and modify.
