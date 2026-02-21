"""
Data Preparation Script for Fine-Tuning
Prepares domain-specific data for instruction fine-tuning
"""

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import List, Dict

def prepare_instruction_dataset(
    data: List[Dict[str, str]],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output"
) -> Dataset:
    """
    Prepare data in instruction format.
    
    Expected format for each item:
    {
        "instruction": "What is the statute of limitations for breach of contract?",
        "input": "",  # Optional context
        "output": "In most jurisdictions, the statute of limitations..."
    }
    """
    
    formatted_data = []
    
    for item in data:
        instruction = item.get(instruction_key, "")
        input_text = item.get(input_key, "")
        output = item.get(output_key, "")
        
        # Format as instruction-following prompt
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""
        
        formatted_data.append({
            "text": prompt,
            "instruction": instruction,
            "output": output
        })
    
    return Dataset.from_list(formatted_data)


def load_from_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_from_csv(filepath: str) -> List[Dict]:
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    return df.to_dict('records')


def create_medical_example_data():
    """Example: Create sample medical Q&A data"""
    examples = [
        {
            "instruction": "What are the common symptoms of Type 2 diabetes?",
            "input": "",
            "output": "Common symptoms of Type 2 diabetes include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. However, Type 2 diabetes often develops slowly, and symptoms may be mild or absent in early stages."
        },
        {
            "instruction": "Explain the mechanism of action for metformin.",
            "input": "",
            "output": "Metformin works primarily by decreasing hepatic glucose production and improving insulin sensitivity in peripheral tissues. It activates AMP-activated protein kinase (AMPK), which leads to reduced gluconeogenesis in the liver and enhanced glucose uptake in muscles. Unlike sulfonylureas, it does not stimulate insulin secretion."
        },
        # Add more examples...
    ]
    return examples


def create_legal_example_data():
    """Example: Create sample legal Q&A data"""
    examples = [
        {
            "instruction": "What constitutes consideration in contract law?",
            "input": "",
            "output": "Consideration is something of value exchanged between parties in a contract. It can be a promise, an act, or forbearance. For a contract to be valid, consideration must be: (1) bargained for, (2) have legal value, and (3) be exchanged between parties. Past consideration typically does not count."
        },
        {
            "instruction": "Analyze this contract clause for potential issues.",
            "input": "The party agrees to waive all rights to dispute resolution and accepts unlimited liability for any damages.",
            "output": "This clause raises several red flags: (1) Blanket waiver of dispute resolution rights may be unenforceable in many jurisdictions, (2) 'Unlimited liability' is unusually broad and potentially unconscionable, (3) Lack of consideration for such waivers, (4) May violate public policy. Recommend significant revision with specific limitations."
        },
        # Add more examples...
    ]
    return examples


def validate_dataset(dataset: Dataset) -> Dict[str, any]:
    """Validate dataset quality"""
    stats = {
        "total_examples": len(dataset),
        "avg_response_length": sum(len(item["output"]) for item in dataset) / len(dataset),
        "empty_outputs": sum(1 for item in dataset if not item["output"]),
        "empty_instructions": sum(1 for item in dataset if not item["instruction"])
    }
    return stats


# Example usage
if __name__ == "__main__":
    print("=== Data Preparation Example ===\n")
    
    # Option 1: Create example data (for testing)
    print("Creating example medical data...")
    medical_data = create_medical_example_data()
    dataset = prepare_instruction_dataset(medical_data)
    
    # Option 2: Load from your own files
    # data = load_from_jsonl("data/train.jsonl")
    # dataset = prepare_instruction_dataset(data)
    
    # Validate
    stats = validate_dataset(dataset)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save prepared dataset
    print("\nSaving dataset...")
    dataset.save_to_disk("data/prepared_dataset")
    
    # Or create train/validation split
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict.save_to_disk("data/prepared_dataset_split")
    
    print("\n✓ Data preparation complete!")
    print("\nNext step: Run train.py to start fine-tuning")
