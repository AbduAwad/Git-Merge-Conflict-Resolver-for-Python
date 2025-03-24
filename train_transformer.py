import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

MODEL_CONFIG = {
    "model_name": "Salesforce/codet5-base",
    "max_input_length": 768,
    "max_output_length": 768,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-5,
    "num_epochs": 5,
    "output_dir": "./model_output_codet5_synthetic_25000",
    "fp16": True,
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
        input_text = (
            f"Resolve the following merge conflict:\n"
            f"<ORIGINAL>\n{example['original']}\n</ORIGINAL>\n"
            f"<BRANCH_A>\n{example['branch_a']}\n</BRANCH_A>\n"
            f"<BRANCH_B>\n{example['branch_b']}\n</BRANCH_B>\n"
            f"<MERGED>"
        )

        output_text = example['merged']

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
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels.long()
        }

def custom_data_collator(features):
    # Use HuggingFace's default collator
    batch = default_data_collator(features)

    # Convert labels to tensor efficiently
    if isinstance(batch["labels"], list):
        batch["labels"] = torch.tensor(np.array(batch["labels"]), dtype=torch.int64)

    return batch

def prepare_dataset(json_path="/content/drive/MyDrive/synthetic_dataset/synthetic_merge_conflicts_50000_batched.json", max_samples=25000):
    import json

    if not os.path.exists(json_path):
        print(f"❌ JSON dataset file {json_path} not found.")
        return [], [], []

    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        print("❌ No data found in the JSON file.")
        return [], [], []

    if len(data) < max_samples:
        print(f"⚠️ Only {len(data)} samples available, less than requested {max_samples}")

    examples = data[:max_samples]

    train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)

    print(f"✅ Loaded from JSON. Train={len(train_examples)}, Val={len(val_examples)}, Test={len(test_examples)}")
    return train_examples, val_examples, test_examples

def process_file_content(content):
    content = re.sub(r'\n\s*\n', '\n\n', content)
    return content.replace('\r\n', '\n').strip()

def train_model():
    try:
        print("🔄 Loading tokenizer and model...")
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CONFIG["model_name"])
        model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG["model_name"])
        model.config.use_cache = False
        model.to(device)

        print("📚 Preparing datasets...")
        train_examples, val_examples, test_examples = prepare_dataset()
        if len(train_examples) == 0:
            print("❌ No training data. Exiting.")
            return None, None, []

        train_dataset = MergeConflictDataset(train_examples, tokenizer, MODEL_CONFIG["max_input_length"], MODEL_CONFIG["max_output_length"])
        val_dataset = MergeConflictDataset(val_examples, tokenizer, MODEL_CONFIG["max_input_length"], MODEL_CONFIG["max_output_length"])

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=MODEL_CONFIG["max_input_length"]
        )

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
            gradient_checkpointing=True,
            report_to=[],
        )

        trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=val_dataset,
          data_collator=custom_data_collator,  # ✅ Replace this
        )


        print("🚀 Starting training...")
        trainer.train()

        model.save_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))
        tokenizer.save_pretrained(os.path.join(MODEL_CONFIG["output_dir"], "final_model"))

        try:
            drive_output_path = "/content/drive/MyDrive/model_output_codet5_synthetic_25000"
            os.makedirs(drive_output_path, exist_ok=True)
            import shutil
            shutil.copytree(
                os.path.join(MODEL_CONFIG["output_dir"], "final_model"),
                os.path.join(drive_output_path, "final_model"),
                dirs_exist_ok=True
            )
            print(f"✅ Model saved to Google Drive at {drive_output_path}/final_model")
        except Exception as e:
            print(f"⚠️ Failed to save model to Google Drive: {e}")

        return model, tokenizer, test_examples

    except Exception as e:
        import traceback
        print(f"❌ Error in training: {e}")
        print(traceback.format_exc())
        return None, None, []

def main():
    try:
        model_path = os.path.join(MODEL_CONFIG["output_dir"], "final_model")
        print(f"🔍 Checking for existing model at {model_path}")
        if os.path.exists(model_path):
            print("📦 Loading existing model...")
            tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            model.to(device)
            _, _, test_examples = prepare_dataset()
        else:
            print("🆕 Training new model...")
            model, tokenizer, test_examples = train_model()

        if model and tokenizer and test_examples:
            example = test_examples[0]
            merged = apply_model_to_conflict(model, tokenizer, example["original"], example["branch_a"], example["branch_b"])
            print("\n🧪 Sample Prediction:")
            print("Predicted Merge:\n", merged)
            print("Actual Merge:\n", example["merged"])

    except Exception as e:
        import traceback
        print(f"❌ Error in main function: {e}")
        print(traceback.format_exc())

def apply_model_to_conflict(model, tokenizer, original, branch_a, branch_b):
    input_text = (
        f"Resolve the following merge conflict:\n"
        f"<ORIGINAL>\n{original}\n</ORIGINAL>\n"
        f"<BRANCH_A>\n{branch_a}\n</BRANCH_A>\n"
        f"<BRANCH_B>\n{branch_b}\n</BRANCH_B>\n"
        f"<MERGED>"
    )

    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=768,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=768,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    merged = tokenizer.decode(outputs[0], skip_special_tokens=True)
    merged = re.sub(r"</?MERGED>", "", merged).strip()
    return merged

if __name__ == "__main__":
    main()
