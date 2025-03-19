import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import re
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model Configuration - CPU optimized
MODEL_CONFIG = {
    "model_name": "facebook/bart-base",  # Smaller model variant for CPU
    "max_input_length": 768,  # Adjust based on your code sizes
    "max_output_length": 768,
    "batch_size": 4,  # Small batch size for CPU
    "gradient_accumulation_steps": 4,  # Simulate larger batch sizes
    "learning_rate": 5e-5,
    "num_epochs": 5,
    "output_dir": "./model_output",
    "fp16": False,  # Set to True only if your CPU supports it
}

class MergeConflictDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input_length, max_output_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format input with special tokens to delineate sections
        input_text = f"<O>\n{example['original']}\n</O>\n<A>\n{example['branch_a']}\n</A>\n<B>\n{example['branch_b']}\n</B>"
        output_text = example['merged']
        
        # Tokenize inputs and outputs
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        output_encodings = self.tokenizer(
            output_text,
            truncation=True,
            max_length=self.max_output_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = input_encodings["input_ids"].squeeze()
        attention_mask = input_encodings["attention_mask"].squeeze()
        labels = output_encodings["input_ids"].squeeze()
        
        # Replace padding token id with -100 to ignore in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_dataset(dataset_dir="dataset/conflicts-py", max_samples=10):
    """
    Loads up to `max_samples` merge conflict instances from the dataset directory.
    Each conflict instance is expected to be a folder containing O.py, A.py, B.py, and M.py.
    """
    examples = []
    
    # Find all subdirectories containing conflict instances
    conflict_folders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]
    
    for folder in conflict_folders[:max_samples]:  # Limit to 20 instances
        try:
            example = {
                "original": open(os.path.join(folder, "O.py")).read(),
                "branch_a": open(os.path.join(folder, "A.py")).read(),
                "branch_b": open(os.path.join(folder, "B.py")).read(),
                "merged": open(os.path.join(folder, "M.py")).read(),
            }
            examples.append(example)
        except Exception as e:
            print(f"Skipping {folder}: {e}")

    if len(examples) < max_samples:
        print(f"⚠️ Warning: Only {len(examples)} samples found. Consider adding more data.")

    # Split into train/validation/test sets
    train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    print(f"✅ Dataset split: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    
    return train_examples, val_examples, test_examples

def process_file_content(content):
    """Clean and normalize file content"""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    # Normalize line endings
    content = content.replace('\r\n', '\n')
    return content.strip()

def parse_example_from_files(original_file, branch_a_file, branch_b_file, merged_file):
    """Parse example from file paths"""
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original = process_file_content(f.read())
        with open(branch_a_file, 'r', encoding='utf-8') as f:
            branch_a = process_file_content(f.read())
        with open(branch_b_file, 'r', encoding='utf-8') as f:
            branch_b = process_file_content(f.read())
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged = process_file_content(f.read())
            
        return {
            "original": original,
            "branch_a": branch_a,
            "branch_b": branch_b,
            "merged": merged
        }
    except Exception as e:
        logger.error(f"Error parsing files: {str(e)}")
        return None

def augment_dataset(examples, num_augmentations=5):
    """
    Augment dataset by applying simple transformations
    This helps with limited data scenarios
    """
    augmented_examples = []
    
    for example in examples:
        augmented_examples.append(example)  # Keep original
        
        # Apply simple augmentations
        for i in range(num_augmentations):
            # Augmentation 1: Randomly change whitespace
            aug_example = {
                "original": example["original"],
                "branch_a": re.sub(r'(\s+)', lambda m: ' ' if m.group(0).isspace() else m.group(0), example["branch_a"]),
                "branch_b": re.sub(r'(\s+)', lambda m: ' ' if m.group(0).isspace() else m.group(0), example["branch_b"]),
                "merged": example["merged"]
            }
            augmented_examples.append(aug_example)
            
            # Augmentation 2: Swap branch A and B (with probability)
            if np.random.random() > 0.5:
                aug_example = {
                    "original": example["original"],
                    "branch_a": example["branch_b"],
                    "branch_b": example["branch_a"],
                    "merged": example["merged"]
                }
                augmented_examples.append(aug_example)
    
    return augmented_examples

def train_model():
    """Train the merge conflict resolution model with CPU optimizations"""
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    
    # Add special tokens for code sections
    special_tokens = {"additional_special_tokens": ["<O>", "</O>", "<A>", "</A>", "<B>", "</B>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model - using BART-base for CPU efficiency
    model = BartForConditionalGeneration.from_pretrained(MODEL_CONFIG["model_name"])
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to device
    model.to(device)
    
    # Prepare datasets
    train_examples, val_examples, test_examples = prepare_dataset()
    
    # Augment training data (helpful for limited datasets)
    if len(train_examples) < 100:  # Only augment if dataset is small
        train_examples = augment_dataset(train_examples)
        logger.info(f"Augmented training data: {len(train_examples)} examples")
    
    # Create dataset objects
    train_dataset = MergeConflictDataset(
        train_examples, 
        tokenizer, 
        MODEL_CONFIG["max_input_length"], 
        MODEL_CONFIG["max_output_length"]
    )
    
    val_dataset = MergeConflictDataset(
        val_examples,
        tokenizer,
        MODEL_CONFIG["max_input_length"],
        MODEL_CONFIG["max_output_length"]
    )
    
    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=MODEL_CONFIG["max_input_length"]
    )
    
    # Set up training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=MODEL_CONFIG["output_dir"],
        num_train_epochs=MODEL_CONFIG["num_epochs"],
        per_device_train_batch_size=MODEL_CONFIG["batch_size"],
        per_device_eval_batch_size=MODEL_CONFIG["batch_size"],
        gradient_accumulation_steps=MODEL_CONFIG["gradient_accumulation_steps"],
        learning_rate=MODEL_CONFIG["learning_rate"],
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        fp16=MODEL_CONFIG["fp16"],
        # Additional CPU optimizations
        dataloader_num_workers=0,  # Prevents multiprocessing issues on some systems
        optim="adamw_torch",  # Use PyTorch's implementation which is more CPU-friendly
        gradient_checkpointing=True,  # Reduces memory usage at the cost of computation time
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting model training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
    tokenizer.save_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
    
    return model, tokenizer, test_examples

def evaluate_model(model, tokenizer, test_examples):
    """Evaluate the trained model on test data"""
    model.eval()
    
    # Create test dataset
    test_dataset = MergeConflictDataset(
        test_examples,
        tokenizer,
        MODEL_CONFIG["max_input_length"],
        MODEL_CONFIG["max_output_length"]
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=MODEL_CONFIG["batch_size"],
        shuffle=False
    )
    
    results = []
    
    # Evaluate model on test data
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate predictions with beam search
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MODEL_CONFIG["max_output_length"],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Store results
            for i, pred in enumerate(predictions):
                idx = i + len(results)
                example = test_examples[idx % len(test_examples)]  # Handle case where batch size doesn't divide evenly
                
                # Calculate simple metrics
                exact_match = pred.strip() == example["merged"].strip()
                
                results.append({
                    "predicted_merge": pred,
                    "actual_merge": example["merged"],
                    "exact_match": exact_match
                })
    
    # Calculate overall metrics
    exact_match_rate = sum(r["exact_match"] for r in results) / len(results)
    logger.info(f"Exact match rate: {exact_match_rate:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_predictions.csv", index=False)
    
    return results_df

def apply_model_to_conflict(model, tokenizer, original, branch_a, branch_b):
    """Apply the trained model to resolve a new conflict"""
    # Format input with special tokens
    input_text = f"<O>\n{original}\n</O>\n<A>\n{branch_a}\n</A>\n<B>\n{branch_b}\n</B>"
    
    # Tokenize input
    input_encodings = tokenizer(
        input_text,
        truncation=True,
        max_length=MODEL_CONFIG["max_input_length"],
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = input_encodings["input_ids"].to(device)
    attention_mask = input_encodings["attention_mask"].to(device)
    
    # Generate merge resolution
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=MODEL_CONFIG["max_output_length"],
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    # Decode prediction
    merged = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return merged

def main():
    """Main function to run the pipeline"""
    # Check if we need to train a new model or load an existing one
    if os.path.exists(os.path.join(MODEL_CONFIG["output_dir"], "final_model")):
        # Load existing model
        logger.info("Loading existing model...")
        tokenizer = BartTokenizer.from_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
        model = BartForConditionalGeneration.from_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
        model.to(device)
        
        # Load test examples
        _, _, test_examples = prepare_dataset()
    else:
        # Train new model
        logger.info("Training new model...")
        model, tokenizer, test_examples = train_model()
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_examples)
    print(results)
    
    # Demo: Apply model to a sample conflict
    if len(test_examples) > 0:
        example = test_examples[0]
        merged = apply_model_to_conflict(
            model, tokenizer, 
            example["original"], 
            example["branch_a"], 
            example["branch_b"]
        )
        
        logger.info("\nExample conflict resolution:")
        logger.info(f"Predicted merge:\n{merged}")
        logger.info(f"Actual merge:\n{example['merged']}")
    
    logger.info("Pipeline complete!")

if __name__ == "__main__":
    main()