import os
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm
import evaluate
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np

# Config
MODEL_PATH = "/content/model_output_codet5_1000/final_model"
TEST_DATASET_DIR = "/content/drive/MyDrive/dataset/conflicts-py"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer + model
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

# === Dataset class ===
class MergeConflictDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input_length=768, max_output_length=768):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = f"merge conflict: original = {example['original']} branch_a = {example['branch_a']} branch_b = {example['branch_b']}"
        output_text = example["merged"]
        input_encodings = self.tokenizer(input_text, truncation=True, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
        output_encodings = self.tokenizer(output_text, truncation=True, max_length=self.max_output_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": input_encodings["input_ids"].squeeze(),
            "attention_mask": input_encodings["attention_mask"].squeeze(),
            "labels": output_encodings["input_ids"].squeeze()
        }

# === Load test data ===
def process_file_content(content):
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.replace('\r\n', '\n')
    return content.strip()

def load_test_examples(dataset_dir=TEST_DATASET_DIR, max_samples=100):
    examples = []
    folders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]
    for folder in folders[:max_samples]:
        try:
            paths = {name: os.path.join(folder, f"{name}.py") for name in ["O", "A", "B", "M"]}
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            with open(paths["O"], "r", encoding="utf-8", errors="replace") as f:
                original = process_file_content(f.read())
            with open(paths["A"], "r", encoding="utf-8", errors="replace") as f:
                branch_a = process_file_content(f.read())
            with open(paths["B"], "r", encoding="utf-8", errors="replace") as f:
                branch_b = process_file_content(f.read())
            with open(paths["M"], "r", encoding="utf-8", errors="replace") as f:
                merged = process_file_content(f.read())
            examples.append({"original": original, "branch_a": branch_a, "branch_b": branch_b, "merged": merged})
        except Exception:
            continue
    return examples

# === Metric computation ===
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def compute_nlp_metrics(preds, refs):
    preds = [re.sub(r'\s+', ' ', p.strip()) for p in preds]
    refs = [re.sub(r'\s+', ' ', r.strip()) for r in refs]
    exact_matches = [int(p == r) for p, r in zip(preds, refs)]
    return {
        "exact_match": np.mean(exact_matches),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        **rouge.compute(predictions=preds, references=refs)
    }

# === Run evaluation ===
def evaluate_model():
    test_examples = load_test_examples()
    dataset = MergeConflictDataset(test_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)

    all_preds = []
    all_refs = [ex["merged"] for ex in test_examples]

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=768,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(decoded)

    metrics = compute_nlp_metrics(all_preds, all_refs)
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics

if __name__ == "__main__":
    evaluate_model()
