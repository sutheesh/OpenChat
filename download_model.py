"""
Download Qwen 2.5 Model
This script downloads the model before running the app
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model():
    """Downloading Llama 3.3 (8B)"""
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("="*60)
    print("Downloading Qwen 2.5 (1.5B) Instruct Model")
    print("="*60)
    print(f"\nModel: {model_name}")
    print("Size: ~3GB")
    print("This will take 10-15 minutes depending on your internet speed")
    
    try:
        print("Step 1/2: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer downloaded successfully!\n")
        
        print("Step 2/2: Downloading model...")
        print("(You'll see download progress bars below)\n")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("\n✓ Model downloaded successfully!\n")
        
        # Check cache location
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print("="*60)
        print("Download Complete!")
        print("="*60)
        print(f"\nModel cached at: {cache_dir}")
        print(f"Model name: {model_name}")
        print("\nYou can now run: python app.py")
        print("The model will load instantly from cache!")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have ~10GB free disk space")
        print("3. Try running again")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Qwen 2.5 Model Downloader")
    print("="*60 + "\n")
    
    success = download_model()
    
    if success:
        print("\n✅ Ready to use! Run 'python app.py' to start the chat.\n")
    else:
        print("\n❌ Download failed. Please try again.\n")
