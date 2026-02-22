"""
AI Chat with Weather Tool + Confluence RAG + Fixed Streaming
Proper async/sync streaming implementation
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import json
import os
import asyncio
from threading import Lock, Thread
from typing import List, Dict, Tuple, Optional, Generator
from functools import wraps

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_path": None,
    "device": "cpu",
    "max_length": 200,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "tool_temperature": 0.0,
    "max_tool_tokens": 220,
    "max_tool_rounds": 3,
    "apify_token": ""
}

# ============================================================================
# GLOBAL STATE
# ============================================================================
model = None
tokenizer = None
model_lock = Lock()
model_loaded = False

weather_mcp = None
weather_lock = Lock()

confluence_rag = None
confluence_lock = Lock()

# ============================================================================
# ASYNC HELPER
# ============================================================================
def async_route(f):
    """Decorator for async Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_once():
    """Load Qwen model"""
    global model, tokenizer, model_loaded

    if model_loaded:
        return

    with model_lock:
        if model_loaded:
            return

        print("📦 Loading Qwen 2.5-1.5B model...")

        tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            device_map="cpu",
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if CONFIG["adapter_path"] and os.path.exists(CONFIG["adapter_path"]):
            model = PeftModel.from_pretrained(model, CONFIG["adapter_path"])
            model = model.merge_and_unload()

        model.eval()
        model_loaded = True
        print("✓ Model loaded on CPU")

# ============================================================================
# WEATHER MCP
# ============================================================================
def get_weather_mcp():
    """Get or initialize Apify Weather MCP client"""
    global weather_mcp

    if weather_mcp is not None:
        return weather_mcp

    with weather_lock:
        if weather_mcp is not None:
            return weather_mcp

        from apify_weather_mcp import ApifyWeatherMCP

        print("🌤️  Initializing Apify Weather MCP...")
        weather_mcp = ApifyWeatherMCP(apify_token=CONFIG["apify_token"])

        return weather_mcp

# ============================================================================
# CONFLUENCE RAG  (NEW)
# ============================================================================
def get_confluence_rag():
    """Get or initialize Confluence RAG engine (singleton)"""
    global confluence_rag

    if confluence_rag is not None:
        return confluence_rag

    with confluence_lock:
        if confluence_rag is not None:
            return confluence_rag

        from confluence_rag import ConfluenceRAG

        print("📚 Initializing Confluence RAG...")
        confluence_rag = ConfluenceRAG()
        count = confluence_rag.collection.count()
        print(f"✓ Confluence RAG ready — {count} chunks indexed")

        if count == 0:
            print("⚠️  WARNING: Knowledge base is empty! Run: python ingest.py")

        return confluence_rag

# ============================================================================
# UTILITIES
# ============================================================================
def extract_json_object(text: str) -> Optional[str]:
    """Extract first valid JSON object"""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def is_valid_tool_call(payload: dict) -> bool:
    """Validate tool call format"""
    if not isinstance(payload, dict):
        return False
    tool_call = payload.get("tool_call")
    if not isinstance(tool_call, dict):
        return False
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    return isinstance(name, str) and name.strip() and isinstance(args, dict)

# ============================================================================
# GENERATION
# ============================================================================
def build_chat_prompt(messages: List[Dict]) -> str:
    """Apply Qwen chat template"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_text(prompt: str, max_tokens: int, temperature: float, do_sample: bool) -> str:
    """Generate text (non-streaming)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=CONFIG["top_p"] if do_sample else 1.0,
            top_k=CONFIG["top_k"] if do_sample else 0,
            repetition_penalty=CONFIG["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def generate_streaming(prompt: str) -> Generator[str, None, None]:
    """Generate text with streaming"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=CONFIG["max_length"],
        temperature=CONFIG["temperature"],
        do_sample=True,
        top_p=CONFIG["top_p"],
        top_k=CONFIG["top_k"],
        repetition_penalty=CONFIG["repetition_penalty"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token

# ============================================================================
# TOOL CALLING  (UPDATED — now includes search_confluence)
# ============================================================================
def decide_tool_or_answer(conversation: List[Dict]) -> Tuple[str, Optional[Dict]]:
    """Model decides: call tool OR answer directly"""
    system_prompt = """You are a helpful assistant with access to two tools.

TOOLS AVAILABLE:
1. get_weather(city)        → Get current weather for a city
2. search_confluence(query) → Search the internal knowledge base for documentation, guides, integrations, how-to articles, and internal content

STRICT RULES — follow exactly:
- If user asks about weather, temperature, forecast → use get_weather
- If user asks about documentation, guides, how to do something, integrations, processes, or ANY internal knowledge → use search_confluence IMMEDIATELY. Do NOT answer from memory.
- NEVER say "I don't have information" without first trying search_confluence
- Tool calls MUST use this exact JSON format and nothing else:

For weather:
{"tool_call": {"name": "get_weather", "arguments": {"city": "CityName"}}}

For knowledge base:
{"tool_call": {"name": "search_confluence", "arguments": {"query": "search query here"}}}

When calling a tool: Output ONLY the JSON above, no extra text.
When answering directly (no tool needed): Provide a clear response, no JSON.
"""

    messages = [{"role": "system", "content": system_prompt}] + conversation
    prompt = build_chat_prompt(messages)

    raw_output = generate_text(
        prompt,
        max_tokens=CONFIG["max_tool_tokens"],
        temperature=CONFIG["tool_temperature"],
        do_sample=False
    )

    print(f"  🔍 DEBUG model decision: {repr(raw_output)}")  # helpful for debugging

    json_str = extract_json_object(raw_output)
    if json_str:
        try:
            payload = json.loads(json_str)
            if is_valid_tool_call(payload):
                return raw_output, payload["tool_call"]
        except json.JSONDecodeError:
            pass

    return raw_output, None


def generate_acknowledgment(tool_name: str, tool_args: dict) -> str:
    """Generate natural acknowledgment before calling any tool"""
    if tool_name == "get_weather":
        city = tool_args.get("city", "")
        context = f"checking weather for {city}"
    else:
        query = tool_args.get("query", "")
        context = f"searching the knowledge base for: {query}"

    system_prompt = f"""You are about to {context}.

Generate a brief, natural acknowledgment (1 sentence) before fetching data.

Examples:
- "Let me check that for you..."
- "I'll look that up right now..."
- "Searching the knowledge base for you..."

Be natural and conversational. One sentence only."""

    messages = [{"role": "system", "content": system_prompt}]
    prompt = build_chat_prompt(messages)

    ack = generate_text(prompt, max_tokens=50, temperature=0.8, do_sample=True)
    return ack.strip()


def call_tool(tool_name: str, tool_args: dict) -> str:
    """Execute any tool and return result as string"""

    # ── Weather ──────────────────────────────────────────────────────────────
    if tool_name == "get_weather":
        city = tool_args.get("city", "").strip()
        if not city:
            return "Error: city name is required"

        mcp_client = get_weather_mcp()
        tool_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(tool_loop)
        try:
            weather_data = tool_loop.run_until_complete(mcp_client.get_weather(city))
            return str(weather_data)
        finally:
            tool_loop.close()

    # ── Confluence RAG (NEW) ─────────────────────────────────────────────────
    elif tool_name == "search_confluence":
        query = tool_args.get("query", "").strip()
        if not query:
            return "Error: query is required"

        rag = get_confluence_rag()
        results = rag.search(query, top_k=5)
        context = rag.format_context(results)
        print(f"  📚 Found {len(results)} chunks for: '{query}'")
        return context

    else:
        return f"Error: Unknown tool '{tool_name}'"

# ============================================================================
# MAIN CHAT LOGIC
# ============================================================================
def process_chat_streaming(user_message: str) -> Generator[str, None, None]:
    """
    Process chat with streaming.
    Runs tool calls synchronously, streams acknowledgment and final answer.
    """
    conversation = [{"role": "user", "content": user_message}]

    for round_num in range(CONFIG["max_tool_rounds"]):
        raw_output, tool_call = decide_tool_or_answer(conversation)

        if tool_call is None:
            # Direct answer — stream it
            for char in raw_output:
                yield char
            return

        tool_name = tool_call["name"].strip()
        tool_args = tool_call.get("arguments", {}) or {}

        if tool_name not in ("get_weather", "search_confluence"):
            error_msg = f"Sorry, I don't have a '{tool_name}' tool available."
            for char in error_msg:
                yield char
            return

        # Stream acknowledgment
        print(f"  💬 Generating acknowledgment for {tool_name}...")
        ack = generate_acknowledgment(tool_name, tool_args)
        ack = ack.strip().strip('"').strip("'")   # remove quotes model adds
        for char in ack:
            yield char
        yield "\n\n[TOOL_EXECUTING]\n\n" 

        # Execute tool
        print(f"  🔧 Calling tool: {tool_name} with args: {tool_args}")
        try:
            tool_result = call_tool(tool_name, tool_args)
        except Exception as e:
            print(f"  ✗ Tool error: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            for char in error_msg:
                yield char
            return

        # Add tool result to conversation
        conversation.append({
            "role": "assistant",
            "content": f"[Called {tool_name} with {tool_args}]"
        })
        conversation.append({
            "role": "user",
            "content": f"[Tool result]:\n{tool_result}"
        })

        print(f"  ✓ Tool result received")

    # Final streaming answer
    print(f"  💬 Streaming final answer...")

    system_prompt = """You are a helpful assistant.
Use the tool results to answer the user's question clearly and naturally.
If the tool returned source URLs or links, include them in your answer.
Don't mention tool names or technical details."""

    messages = [{"role": "system", "content": system_prompt}] + conversation
    prompt = build_chat_prompt(messages)

    for token in generate_streaming(prompt):
        yield token


def process_chat_non_streaming(user_message: str) -> str:
    """Process chat without streaming"""
    conversation = [{"role": "user", "content": user_message}]

    for round_num in range(CONFIG["max_tool_rounds"]):
        raw_output, tool_call = decide_tool_or_answer(conversation)

        if tool_call is None:
            return raw_output

        tool_name = tool_call["name"].strip()
        tool_args = tool_call.get("arguments", {}) or {}

        if tool_name not in ("get_weather", "search_confluence"):
            return f"Sorry, I don't have a '{tool_name}' tool available."

        print(f"  🔧 Calling tool: {tool_name} with args: {tool_args}")
        try:
            tool_result = call_tool(tool_name, tool_args)
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

        conversation.append({
            "role": "assistant",
            "content": f"[Called {tool_name} with {tool_args}]"
        })
        conversation.append({
            "role": "user",
            "content": f"[Tool result]:\n{tool_result}"
        })

    # Final answer
    system_prompt = """You are a helpful assistant.
Use the tool results to answer the user's question clearly and naturally.
If the tool returned source URLs or links, include them in your answer."""

    messages = [{"role": "system", "content": system_prompt}] + conversation
    prompt = build_chat_prompt(messages)

    return generate_text(prompt, CONFIG["max_length"], CONFIG["temperature"], True)

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    """Serve UI"""
    load_model_once()
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
@async_route
async def api_chat():
    """Main chat endpoint with streaming support"""
    try:
        load_model_once()

        data = request.json or {}
        message = (data.get('message') or '').strip()
        stream = bool(data.get('stream', False))

        if not message:
            return jsonify({'error': 'Message required'}), 400

        if stream:
            def generate():
                try:
                    for token in process_chat_streaming(message):
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as e:
                    print(f"  ✗ Streaming error: {e}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

        else:
            response = process_chat_non_streaming(message)
            return jsonify({
                'response': response,
                'model': 'Qwen 2.5-1.5B'
            })

    except Exception as e:
        print(f"✗ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Server status"""
    rag_count = 0
    try:
        rag = get_confluence_rag()
        rag_count = rag.collection.count()
    except Exception:
        pass

    return jsonify({
        'model_loaded': model_loaded,
        'model_name': CONFIG['base_model'],
        'device': 'CPU (Mac Intel)',
        'streaming_supported': True,
        'confluence_chunks': rag_count   # NEW: shows KB health
    })


# NEW: Manual KB refresh endpoint
@app.route('/api/refresh-kb', methods=['POST'])
def refresh_kb():
    """Re-index Confluence knowledge base on demand"""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "ingest.py", "--reset"],
            capture_output=True, text=True, timeout=300
        )
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout[-2000:],  # last 2000 chars
            'error': result.stderr[-500:] if result.stderr else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🤖 AI Chat with Weather + Confluence RAG + Streaming")
    print("=" * 70)

    print(f"\n📦 Model: {CONFIG['base_model']}")
    print(f"💻 Device: CPU")
    print(f"🌤️  Weather: Apify MCP")
    print(f"📚 Confluence: RAG (ChromaDB)")
    print(f"📡 Streaming: Enabled")

    print("\n" + "=" * 70)
    print("🚀 Server at http://localhost:8000")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)