"""
Fine-Tuning Script using LoRA/QLoRA
Efficiently fine-tune large language models on a single GPU
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import os

# ============================================================================
# CONFIGURATION - Adjust these parameters for your use case
# ============================================================================

CONFIG = {
    # Model selection
    "base_model": "mistralai/Mistral-7B-v0.1",  # or "meta-llama/Llama-2-7b-hf", "gpt2-xl"
    
    # Data
    "dataset_path": "data/prepared_dataset_split",
    
    # LoRA parameters
    "lora_r": 16,  # Rank of the low-rank matrices (higher = more parameters)
    "lora_alpha": 32,  # Scaling factor (typically 2x lora_r)
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to apply LoRA
    
    # Quantization (for memory efficiency)
    "use_4bit": True,  # Use 4-bit quantization (requires less VRAM)
    "bnb_4bit_compute_dtype": "float16",
    
    # Training hyperparameters
    "num_epochs": 3,
    "batch_size": 4,  # Reduce if running out of memory
    "gradient_accumulation_steps": 4,  # Effective batch size = batch_size * gradient_accumulation_steps
    "learning_rate": 2e-4,
    "max_seq_length": 512,  # Maximum sequence length
    "warmup_steps": 100,
    
    # Optimization
    "optim": "paged_adamw_32bit",  # Memory-efficient optimizer
    "fp16": False,  # Use mixed precision (if not using 4-bit)
    "bf16": False,  # Use bfloat16 if your GPU supports it (A100, H100)
    
    # Saving
    "output_dir": "models/fine-tuned-model",
    "save_steps": 100,
    "logging_steps": 10,
}

# ============================================================================
# SETUP
# ============================================================================

def setup_model_and_tokenizer(config):
    """Load and prepare model with quantization and LoRA"""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model...")
    
    # Quantization config (for 4-bit training)
    if config["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    # LoRA configuration
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def preprocess_dataset(dataset, tokenizer, max_length):
    """Tokenize dataset"""
    
    def tokenize_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Labels are the same as input_ids for causal LM
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


# ============================================================================
# TRAINING
# ============================================================================

def train(config):
    """Main training function"""
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(config["dataset_path"])
    
    # Preprocess
    print("Preprocessing dataset...")
    tokenized_dataset = preprocess_dataset(dataset, tokenizer, config["max_seq_length"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["save_steps"],
        evaluation_strategy="steps",
        save_total_limit=3,
        fp16=config["fp16"],
        bf16=config["bf16"],
        optim=config["optim"],
        gradient_checkpointing=True,  # Saves memory
        group_by_length=True,  # Groups similar length sequences
        report_to="tensorboard",  # or "wandb" if you have it configured
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Train!
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {config['output_dir']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
        print("Consider using a GPU or cloud service like Google Colab.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Start training
    train(CONFIG)
    
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Check tensorboard logs: tensorboard --logdir models/fine-tuned-model")
    print("2. Test your model: python inference.py")
    print("="*50)
