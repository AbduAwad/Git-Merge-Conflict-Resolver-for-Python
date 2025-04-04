import os
import numpy as np
import re
from sklearn import logger
from sklearn.model_selection import train_test_split

def process_file_content(content):
    """Clean and normalize file content"""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    # Normalize line endings
    content = content.replace('\r\n', '\n')
    return content.strip()

def prepare_dataset(dataset_dir="../dataset/conflicts-py", max_samples=None):
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
    
    print(f"Dataset split: Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    
    return train_examples, val_examples, test_examples

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