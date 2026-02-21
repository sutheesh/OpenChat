"""
Inference Script - Test your fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def load_model(base_model_path, adapter_path=None):
    """Load base model and optionally apply LoRA adapter"""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load LoRA adapter if provided
    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge LoRA weights into base model
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text="", max_length=512, temperature=0.7):
    """Generate a response to an instruction"""
    
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
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    response = full_response.split("### Response:")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print("\n" + "="*60)
    print("Interactive Mode - Type your questions (or 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        instruction = input("\n🔵 Your question: ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not instruction:
            continue
        
        print("\n🤖 Generating response...\n")
        response = generate_response(model, tokenizer, instruction)
        print(f"📝 Response:\n{response}\n")
        print("-" * 60)


def batch_test(model, tokenizer, test_cases):
    """Test multiple examples"""
    print("\n" + "="*60)
    print("Batch Testing")
    print("="*60 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Instruction: {test['instruction']}")
        if test.get('input'):
            print(f"Input: {test['input']}")
        
        response = generate_response(
            model, 
            tokenizer, 
            test['instruction'], 
            test.get('input', '')
        )
        
        print(f"\nGenerated Response:\n{response}")
        print("\n" + "-"*60)


# Example test cases
MEDICAL_TEST_CASES = [
    {
        "instruction": "What are the primary risk factors for cardiovascular disease?",
        "input": ""
    },
    {
        "instruction": "Explain the difference between Type 1 and Type 2 diabetes.",
        "input": ""
    },
    {
        "instruction": "What medication would you recommend for this patient?",
        "input": "Patient is a 45-year-old male with hypertension (BP 150/95) and no other comorbidities."
    }
]

LEGAL_TEST_CASES = [
    {
        "instruction": "What is the difference between negligence and gross negligence?",
        "input": ""
    },
    {
        "instruction": "Analyze this contract clause for enforceability.",
        "input": "Party A agrees to provide services in perpetuity without compensation."
    },
    {
        "instruction": "What are the elements required to prove breach of contract?",
        "input": ""
    }
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--model_path", type=str, default="models/fine-tuned-model",
                       help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Base model name")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="interactive",
                       help="Run in interactive or batch mode")
    parser.add_argument("--domain", type=str, choices=["medical", "legal", "custom"], default="medical",
                       help="Domain for batch testing")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.base_model, args.model_path)
    print("✓ Model loaded successfully!\n")
    
    # Run in selected mode
    if args.mode == "interactive":
        interactive_mode(model, tokenizer)
    else:
        # Batch mode
        test_cases = MEDICAL_TEST_CASES if args.domain == "medical" else LEGAL_TEST_CASES
        batch_test(model, tokenizer, test_cases)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
