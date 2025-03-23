import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,  # Changed from T5TokenizerFast
    T5ForConditionalGeneration,
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

# Model Configuration - CodeT5 optimized
MODEL_CONFIG = {
    "model_name": "Salesforce/codet5-base",  # Specialized code model
    "max_input_length": 768,  # Adjust based on your code sizes
    "max_output_length": 768,
    "batch_size": 4,  # Small batch size for CPU
    "gradient_accumulation_steps": 4,  # Simulate larger batch sizes
    "learning_rate": 3e-5,  # Slightly lower learning rate for CodeT5
    "num_epochs": 5,
    "output_dir": "./model_output_codet5_5000",
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
        input_text = f"merge conflict: original = {example['original']} branch_a = {example['branch_a']} branch_b = {example['branch_b']}"
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

def prepare_dataset(dataset_dir="/content/drive/MyDrive/dataset/conflicts-py", max_samples=5000):
    """
    Loads up to `max_samples` merge conflict instances from the dataset directory.
    Each conflict instance is expected to be a folder containing O.py, A.py, B.py, and M.py.
    """
    examples = []
    
    # Verify if dataset directory exists
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        return [], [], []
        
    # Find all subdirectories containing conflict instances
    try:
        conflict_folders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]
    except Exception as e:
        logger.error(f"Error scanning dataset directory: {e}")
        return [], [], []
    
    logger.info(f"Found {len(conflict_folders)} potential conflict folders")
    
    for folder in conflict_folders[:max_samples]:
        try:
            # Check if all required files exist
            required_files = ["O.py", "A.py", "B.py", "M.py"]
            if not all(os.path.exists(os.path.join(folder, f)) for f in required_files):
                logger.warning(f"Skipping {folder}: missing required files")
                continue
                
            example = {
                "original": open(os.path.join(folder, "O.py")).read(),
                "branch_a": open(os.path.join(folder, "A.py")).read(),
                "branch_b": open(os.path.join(folder, "B.py")).read(),
                "merged": open(os.path.join(folder, "M.py")).read(),
            }
            
            # Process file content to clean and normalize
            for key in example:
                example[key] = process_file_content(example[key])
                
            examples.append(example)
        except Exception as e:
            logger.warning(f"Skipping {folder}: {e}")

    if len(examples) == 0:
        logger.error("No valid examples found in the dataset")
        return [], [], []
            
    if len(examples) < max_samples:
        logger.warning(f"Only {len(examples)} samples found. Consider adding more data.")

    # Split into train/validation/test sets
    train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    logger.info(f"Dataset split: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    
    return train_examples, val_examples, test_examples

def process_file_content(content):
    """Clean and normalize file content"""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    # Normalize line endings
    content = content.replace('\r\n', '\n')
    return content.strip()

def augment_dataset(examples, num_augmentations=3):
    """
    Augment dataset by applying transformations more suitable for code
    """
    augmented_examples = []
    
    for example in examples:
        augmented_examples.append(example)  # Keep original
        
        # Apply code-specific augmentations
        for i in range(num_augmentations):
            # Augmentation 1: Modify whitespace and indentation
            aug_example = {
                "original": example["original"],
                "branch_a": re.sub(r'(\s+)', lambda m: ' ' * np.random.randint(1, 3) if m.group(0).isspace() else m.group(0), example["branch_a"]),
                "branch_b": re.sub(r'(\s+)', lambda m: ' ' * np.random.randint(1, 3) if m.group(0).isspace() else m.group(0), example["branch_b"]),
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
            
            # Augmentation 3: Add or remove comments
            if np.random.random() > 0.7:
                lines_a = example["branch_a"].split('\n')
                lines_b = example["branch_b"].split('\n')
                
                # Randomly add comments to non-comment lines
                for i in range(len(lines_a)):
                    if not lines_a[i].strip().startswith('#') and np.random.random() > 0.8:
                        lines_a[i] = lines_a[i] + "  # modified line"
                
                for i in range(len(lines_b)):
                    if not lines_b[i].strip().startswith('#') and np.random.random() > 0.8:
                        lines_b[i] = lines_b[i] + "  # modified line"
                
                aug_example = {
                    "original": example["original"],
                    "branch_a": '\n'.join(lines_a),
                    "branch_b": '\n'.join(lines_b),
                    "merged": example["merged"]
                }
                augmented_examples.append(aug_example)
    
    return augmented_examples

def train_model():
    """Train the merge conflict resolution model with CodeT5"""
    try:
        # Load tokenizer - Using RobertaTokenizer for CodeT5
        logger.info(f"Loading tokenizer from {MODEL_CONFIG['model_name']}...")
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CONFIG["model_name"])
        
        # Load model
        logger.info(f"Loading model from {MODEL_CONFIG['model_name']}...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG["model_name"])
        
        # Move model to device
        model.to(device)
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_examples, val_examples, test_examples = prepare_dataset()
        
        if len(train_examples) == 0:
            logger.error("No training examples found. Cannot proceed with training.")
            return None, None, []
        
        # Augment training data (helpful for limited datasets)
        if len(train_examples) < 100:  # Only augment if dataset is small
            logger.info("Augmenting training data...")
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
            padding="max_length",
            max_length=MODEL_CONFIG["max_input_length"]
        )
        
        # Set up training arguments optimized for code models
        training_args = TrainingArguments(
            output_dir=MODEL_CONFIG["output_dir"],
            num_train_epochs=MODEL_CONFIG["num_epochs"],
            per_device_train_batch_size=MODEL_CONFIG["batch_size"],
            per_device_eval_batch_size=MODEL_CONFIG["batch_size"],
            gradient_accumulation_steps=MODEL_CONFIG["gradient_accumulation_steps"],
            learning_rate=MODEL_CONFIG["learning_rate"],
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=10,
            fp16=MODEL_CONFIG["fp16"], 
            gradient_checkpointing=True,  # Reduces memory usage at the cost of computation time
            report_to=[],  # Disable Weights & Biases logging
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
        logger.info("Starting CodeT5 model training...")
        trainer.train()
        
        # Save model
        model.save_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
        tokenizer.save_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))

                # Optional: Save to Google Drive
        try:
            drive_output_path = "/content/drive/MyDrive/codet5_model_output_5000"
            os.makedirs(drive_output_path, exist_ok=True)

            import shutil
            shutil.copytree(
                os.path.join(MODEL_CONFIG["output_dir"], "final_model"),
                os.path.join(drive_output_path, "final_model"),
                dirs_exist_ok=True
            )

            logger.info(f"Model successfully saved to Google Drive at {drive_output_path}/final_model")

        except Exception as e:
            logger.error(f"Failed to save model to Google Drive: {e}")

        
        return model, tokenizer, test_examples
        
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, []



def calculate_token_match(pred_tokens, actual_tokens):
    """Calculate a simple token matching metric"""
    if not pred_tokens or not actual_tokens:
        return 0.0
    
    matches = 0
    for token in pred_tokens:
        if token in actual_tokens:
            matches += 1
    
    return matches / max(len(pred_tokens), len(actual_tokens))

def apply_model_to_conflict(model, tokenizer, original, branch_a, branch_b):
    """Apply the trained model to resolve a new conflict"""
    if not model or not tokenizer:
        logger.error("Cannot apply model: missing model or tokenizer")
        return None
        
    # Format input for CodeT5
    input_text = f"merge conflict: original = {original} branch_a = {branch_a} branch_b = {branch_b}"
    
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
    """Main function to run the pipeline with CodeT5"""
    try:
        # Check if we need to train a new model or load an existing one
        if os.path.exists(os.path.join(MODEL_CONFIG["output_dir"], "final_model")):
            # Load existing model
            logger.info("Loading existing CodeT5 model...")
            try:
                tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
                model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
                model.to(device)
                
                # Load test examples
                _, _, test_examples = prepare_dataset()
            except Exception as e:
                logger.error(f"Error loading existing model: {e}")
                logger.info("Will train a new model instead")
                model, tokenizer, test_examples = train_model()
        else:
            # Train new model
            logger.info("Training new CodeT5 model...")
            model, tokenizer, test_examples = train_model()
        
        if not model or not tokenizer:
            logger.error("Failed to initialize or train model. Exiting.")
            return
        
        # Demo: Apply model to a sample conflict
        if len(test_examples) > 0:
            example = test_examples[0]
            merged = apply_model_to_conflict(
                model, tokenizer, 
                example["original"], 
                example["branch_a"], 
                example["branch_b"]
            )
            
            if merged:
                print("Example conflict resolution:")
                print(f"Predicted merge:\n{merged}")
                print(f"Actual merge:\n{example['merged']}")
        
        logger.info("CodeT5 pipeline complete!")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()


# Perforemance: 

"""

trainer = Trainer(
[965/965 3:40:48, Epoch 4/5]
Epoch	Training Loss	Validation Loss
1	0.584500	0.443484
2	0.432800	0.410092
3	0.428500	0.387623
4	0.374900	0.381603
There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'].
Evaluating: 100%|██████████| 215/215 [1:57:07<00:00, 32.68s/it]
exact_match                   0.002331
whitespace_invariant_match    0.002331
token_match_ratio             0.061981

"""