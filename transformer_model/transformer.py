"""
“This script is used to fine-tune our transformer model on our dataset of merge conflicts.  
It uses a sequence-to-sequence transformer model for code generation, a sentence transformer to obtain the semantic similarity in text, 
and a dataset of synthetic merge conflicts to train and evaluate the model. 
"""

"""
“We start by importing the necessary libraries: PyTorch for training and dataset handling, transformers for using CodeT5, 
sentence_transformers for semantic embeddings, ast for syntax parsing, 
and typical utilities like NumPy, JSON, and typing.”"""

import torch
import ast
import numpy as np
import json
from typing import List, Tuple, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

# It loads merge conflict samples from a JSON file and returns individual examples that include the original code,
#  branch A, branch B, and the resolved (merged) code
class MergeConflictDataset(Dataset): # Standard Python Dataset class
    """
    Custom PyTorch Dataset for merge conflict data
    """
    def __init__(self, dataset_path, num_samples=30000):
        """
        Initialize dataset from JSON file

        Args:
            dataset_path (str): Path to JSON dataset
            num_samples (int): Number of samples to use
        """
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            full_data = json.load(f)

        # Use only the first num_samples
        self.data = full_data[:num_samples]

        print(f"Using {len(self.data)} merge conflict samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset

        Returns:
            dict: Contains 'original', 'branch_a', 'branch_b', and 'merged' code
        """
        return self.data[idx]

# Class for resolving merge conflicts using transformer-based models and semantic analysis
# “This is the main class that contains all the logic for preprocessing, training, and inference.”

class AdvancedMergeConflictResolver:
    # initialize the tokenizer and the CodeT5 model for code generation, 
    # a sentence embedding model for semantic similarity, and load the dataset. 
    # The models are automatically moved to GPU if available
    def __init__(self, 
                 model_name="Salesforce/codet5-base", 
                 embedding_model="all-MiniLM-L6-v2",
                 dataset_path="/content/drive/MyDrive/synthetic_dataset/synthetic_merge_conflicts_50000_batched.json"):
        """
        Initialize merge conflict resolver with code generation, embedding models, and dataset

        Args:
            model_name (str): Transformer model for code generation
            embedding_model (str): Model for semantic embedding
            dataset_path (str): Path to merge conflict dataset
        """

        # Initialize tokenizer and code generation model
        print(f"Initializing code generation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Initialize semantic embedding model for semantic similarity
        print(f"Initializing embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Load dataset (first 50 samples) 
        self.dataset = MergeConflictDataset(dataset_path, num_samples=30000)

        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _advanced_preprocessing(self, text: str) -> str:
        """
        Advanced preprocessing with multiple cleaning steps

        Args:
            text (str): Input code text

        Returns:
            str: Preprocessed code text
        """
        # Remove conflict markers
        lines = [
            line for line in text.split('\n')
            if not line.startswith(('<<<<<<', '=======', '>>>>>>>'))
        ]

        # Normalize whitespace and indentation
        cleaned_lines = [line.rstrip() for line in lines]

        # Resolve and consolidate imports
        imports = set()
        non_import_lines = []
        for line in cleaned_lines:
            if line.startswith(('import ', 'from ')):
                if line not in imports:
                    imports.add(line)
            else:
                non_import_lines.append(line)

        # Combine cleaned imports and code
        cleaned_text = '\n'.join(sorted(list(imports)) + non_import_lines)
        return cleaned_text

    def _generate_code_embedding(self, code: str) -> np.ndarray:
        """
        This uses the sentence transformer to generate a 384-dimensional 
        semantic embedding of the input code. 
        It’s robust to syntax variations but captures overall meaning

        Args:
            code (str): Code text

        Returns:
            np.ndarray: Semantic embedding vector
        """
        try:
            return self.embedding_model.encode([code])[0]
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2

    def _compute_semantic_similarity(self, code1: str, code2: str) -> float:
        """
        evaluate how semantically close two code snippets are

        Args:
            code1 (str): First code snippet
            code2 (str): Second code snippet

        Returns:
            float: Semantic similarity score
        """
        embedding1 = self._generate_code_embedding(code1)
        embedding2 = self._generate_code_embedding(code2)

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    def _extract_function_signature(self, code: str) -> str:
        """
        Extract function signature for semantic comparison

        Args:
            code (str): Code text

        Returns:
            str: Function signature or empty string
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function name, arguments, and return annotation
                    return f"{node.name}{ast.unparse(node.args)}"
        except SyntaxError:
            return ""
        return ""

    def _validate_syntax(self, code: str) -> bool:
        """
        Validate code syntax

        Args:
            code (str): Code text

        Returns:
            bool: True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            print("Syntax validation failed")
            return False

    def _evaluate_merge_quality(
        self,
        original: str,
        merged_code: str
    ) -> float:
        """
        Evaluate quality of merged code

        Args:
            original (str): Original code
            merged_code (str): Merged code

        Returns:
            float: Merge quality score
        """
        score = 0.0

        # Syntax validation
        if self._validate_syntax(merged_code):
            score += 0.3

        # Function signature preservation
        original_signature = self._extract_function_signature(original)
        merged_signature = self._extract_function_signature(merged_code)

        if original_signature == merged_signature:
            score += 0.2

        # Semantic similarity
        semantic_score = self._compute_semantic_similarity(original, merged_code)
        score += 0.3 * semantic_score

        # Code complexity metric
        line_count_diff = abs(len(original.split('\n')) - len(merged_code.split('\n')))
        score -= min(0.2, 0.05 * line_count_diff)

        return max(0, min(1.0, score))

    def prepare_dataloader(self, batch_size=16, shuffle=True):
        """
        Prepare DataLoader for training or inference

        Args:
            batch_size (int): Batch size for DataLoader
            shuffle (bool): Whether to shuffle the dataset

        Returns:
            torch.utils.data.DataLoader: Configured DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train(self, epochs=3, learning_rate=5e-5, batch_size=16):
        """
        Fine-tune the model on the merge conflict dataset

        Args:
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for training
            batch_size (int): Batch size for training
        """
        # Prepare optimizer and dataloader
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate) # AdamW is a variant of Adam optimizer
        dataloader = self.prepare_dataloader(batch_size=batch_size) # DataLoader for batching (will load the dataset in batches)

        print(f"Starting training for {epochs} epochs") 
 
        for epoch in range(epochs): # training loop over epochs
            self.model.train() # Set model to training mode
            total_loss = 0

            for batch in dataloader: # Iterate over batches
                # Prepare inputs and labels
                inputs = self.tokenizer( # tokenize the input text
                    [
                        f"Resolve merge conflict between two code branches.\n"
                        f"Original code:\n{orig}\n\n"
                        f"Branch A changes:\n{a}\n\n"
                        f"Branch B changes:\n{b}\n\n"
                        f"Provide the merged code:"
                        for orig, a, b in zip(batch['original'], batch['branch_a'], batch['branch_b'])
                    ],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                labels = self.tokenizer( # tokenize the merged code or the class label
                    batch['merged'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).input_ids.to(self.device)

                # Forward pass
                outputs = self.model(**inputs, labels=labels) # get the outputs of teh encoder-decoder transformer by passing in the inputs and labels
                loss = outputs.loss # loss is the loss function of the model (cross-entropy loss)

                # Train using gradient descent
                optimizer.zero_grad()  # clear previous gradients 
                loss.backward() # compute gradients through backpropagation
                optimizer.step() # update the model parameters, and go to the nexts step in the optimizer (gradient descent)

                total_loss += loss.item() # accumulate the loss

            # Log epoch statistics
            avg_loss = total_loss / len(dataloader) # average loss over the epoch
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        print("Training completed")

import os

def main(): # Runs the training and evaluation, and saves the model to google drive
    # Initialize the merge conflict resolver with dataset
    resolver = AdvancedMergeConflictResolver() 

    resolver.train(epochs=3, learning_rate=5e-5, batch_size=16) # trains the model for 3 epochs

    output_dir = "merge_conflict_model"
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    resolver.model.save_pretrained(final_model_path)
    resolver.tokenizer.save_pretrained(final_model_path)

    try:
        drive_output_path = "/content/drive/MyDrive/merge_conflict_model"
        drive_final_model_path = os.path.join(drive_output_path, "final_model")
        os.makedirs(drive_final_model_path, exist_ok=True)
        import shutil
        shutil.copytree(
            final_model_path,
            drive_final_model_path,
            dirs_exist_ok=True
        )
        print(f"✅ Model saved to Google Drive at {drive_output_path}/final_model")
    except Exception as e:
        print(f"⚠️ Failed to save model to Google Drive: {e}")


if __name__ == "__main__":
    main()