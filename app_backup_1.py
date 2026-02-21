"""
Web UI for Fine-Tuned Model Inference
A Flask-based web interface to interact with your fine-tuned model
"""

from flask import Flask, render_template, request, jsonify, stream_with_context, Response
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from threading import Lock

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_lock = Lock()
model_loaded = False

# Configuration - Qwen 2.5 Chat Model for Mac Intel CPU
CONFIG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_path": None,  # No fine-tuning
    "max_length": 200,  # Tokens per response
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1
}


def load_model_once():
    """Load model only once (lazy loading) - Optimized for Mac Intel CPU"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return
    
    with model_lock:
        if model_loaded:  # Double-check after acquiring lock
            return
        
        print("Loading Qwen 2.5 (1.5B) Chat model... This may take 1-2 minutes.")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model (CPU-optimized for Mac Intel)
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            device_map="cpu",  # Use CPU (Mac Intel doesn't have CUDA)
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Important for 3GB model on Mac
        )
        
        # Skip adapter loading since we're using base model only
        if CONFIG["adapter_path"] and os.path.exists(CONFIG["adapter_path"]):
            print(f"Loading fine-tuned adapter from {CONFIG['adapter_path']}")
            model = PeftModel.from_pretrained(model, CONFIG["adapter_path"])
            model = model.merge_and_unload()
        
        model.eval()
        model_loaded = True
        print("✓ Qwen 2.5 loaded successfully on CPU!")


def generate_chat_response(message, stream=False):
    """Generate chat response using Qwen's chat template"""
    
    # Use Qwen's chat template format
    messages = [
        {"role": "user", "content": message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if stream:
        # Streaming generation
        return generate_stream_simple(inputs, prompt)
    else:
        # Regular generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_length"],
                temperature=CONFIG["temperature"],
                do_sample=True,
                top_p=CONFIG["top_p"],
                top_k=CONFIG["top_k"],
                repetition_penalty=CONFIG["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (exclude the prompt)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Stop at user turn or special tokens
        response = response.split("<|im_start|>")[0]  # Stop at next turn
        response = response.split("User:")[0]  # Stop if it starts generating user text
        response = response.split("\nUser:")[0]
        response = response.split("\n\nUser:")[0]
        
        return response.strip()


def generate_stream_simple(inputs, original_prompt):
    """Stream generation token by token for chat"""
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
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
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the output, but stop at special markers
    for text in streamer:
        # Stop if we hit these markers (model trying to generate user turn)
        if "<|im_start|>" in text or "User:" in text or "\nUser:" in text:
            break
        yield text


def generate_response(instruction, input_text="", stream=False):
    """Generate response from the model"""
    
    # Format prompt
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if stream:
        # Streaming generation
        return generate_stream(inputs, prompt)
    else:
        # Regular generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_length"],
                temperature=CONFIG["temperature"],
                do_sample=True,
                top_p=CONFIG["top_p"],
                top_k=CONFIG["top_k"],
                repetition_penalty=CONFIG["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("### Response:")[-1].strip()
        return response


def generate_stream(inputs, prompt):
    """Stream generation token by token"""
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=CONFIG["max_length"],
        temperature=CONFIG["temperature"],
        do_sample=True,
        top_p=CONFIG["top_p"],
        top_k=CONFIG["top_k"],
        repetition_penalty=CONFIG["repetition_penalty"],
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the output
    for text in streamer:
        yield text


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    load_model_once()
    return render_template('index.html')  # Fixed: using index.html

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Simple chat endpoint - just question and answer"""
    try:
        # Load model if not already loaded
        load_model_once()
        
        data = request.json
        message = data.get('message', '').strip()
        stream = data.get('stream', False)
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if stream:
            # Streaming response
            def generate():
                for token in generate_chat_response(message, stream=True):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Regular response
            response = generate_chat_response(message)
            return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for text generation"""
    try:
        # Load model if not already loaded
        load_model_once()
        
        data = request.json
        instruction = data.get('instruction', '').strip()
        input_text = data.get('input', '').strip()
        stream = data.get('stream', False)
        
        if not instruction:
            return jsonify({'error': 'Instruction is required'}), 400
        
        if stream:
            # Streaming response
            def generate():
                for token in generate_response(instruction, input_text, stream=True):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Regular response
            response = generate_response(instruction, input_text)
            return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update generation configuration"""
    global CONFIG
    
    if request.method == 'GET':
        return jsonify(CONFIG)
    
    elif request.method == 'POST':
        data = request.json
        
        # Update allowed parameters
        if 'temperature' in data:
            CONFIG['temperature'] = float(data['temperature'])
        if 'max_length' in data:
            CONFIG['max_length'] = int(data['max_length'])
        if 'top_p' in data:
            CONFIG['top_p'] = float(data['top_p'])
        if 'top_k' in data:
            CONFIG['top_k'] = int(data['top_k'])
        if 'repetition_penalty' in data:
            CONFIG['repetition_penalty'] = float(data['repetition_penalty'])
        
        return jsonify({'status': 'success', 'config': CONFIG})


@app.route('/api/status')
def api_status():
    """Check if model is loaded"""
    return jsonify({
        'model_loaded': model_loaded,
        'cuda_available': False,  # Mac Intel doesn't have CUDA
        'gpu_name': 'CPU (Intel Mac)',  # Show CPU info
        'context_window': 32768  # Qwen 2.5 context window
    })


@app.route('/api/examples')
def api_examples():
    """Get example prompts based on domain"""
    examples = {
        'medical': [
            {
                'instruction': 'What are the symptoms of Type 2 diabetes?',
                'input': ''
            },
            {
                'instruction': 'Explain the mechanism of action for this medication.',
                'input': 'Metformin'
            },
            {
                'instruction': 'What differential diagnoses should be considered?',
                'input': 'Patient presents with chest pain and shortness of breath.'
            }
        ],
        'legal': [
            {
                'instruction': 'What is consideration in contract law?',
                'input': ''
            },
            {
                'instruction': 'Analyze this contract clause for potential issues.',
                'input': 'Party A waives all rights to dispute resolution.'
            },
            {
                'instruction': 'What are the elements of negligence?',
                'input': ''
            }
        ],
        'general': [
            {
                'instruction': 'Explain this concept in simple terms.',
                'input': 'Quantum entanglement'
            },
            {
                'instruction': 'Write a summary of this topic.',
                'input': 'The history of artificial intelligence'
            }
        ]
    }
    return jsonify(examples)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting AI Chat Assistant - Mac Intel CPU")
    print("="*60)
    
    # Show system info
    print(f"✓ Running on: Intel Mac (CPU)")
    print(f"⚠️  Note: CPU inference is slower than GPU (30-60s per response)")
    
    print(f"\nModel config:")
    print(f"  Base model: {CONFIG['base_model']} (Qwen 2.5 Chat)")
    print(f"  Parameters: 1.5B")
    print(f"  Device: CPU")
    
    print("\n" + "="*60)
    print("Server starting at http://localhost:8000")
    print("="*60 + "\n")
    
    # Run Flask app on port 8000
    app.run(
        host='0.0.0.0',
        port=8000,  # Changed to port 8000
        debug=False,  # Set to True for development
        threaded=True
    )