import sys
import json
import torch
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os


# Path to your trained model
model_path = os.path.join(os.path.dirname(__file__), "merge_conflict_model-20250325T222421Z-001", "merge_conflict_model", "final_model")


tokenizer = AutoTokenizer.from_pretrained(model_path) # Load the tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path) # Load the trained model
model.to("cuda" if torch.cuda.is_available() else "cpu") # Send the model to the appropriate device

embedding_model_name = "all-MiniLM-L6-v2" # Or the one you initialized with
embedding_model = SentenceTransformer(embedding_model_name)


def resolve_merge_conflict(original, branch_a, branch_b):
    
    input_text = (
        f"Resolve merge conflict between two code branches.\n"
        f"Original code:\n{original}\n\n"
        f"Branch A changes:\n{branch_a}\n\n"
        f"Branch B changes:\n{branch_b}\n\n"
        f"Provide the merged code:"
    )

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    resolved_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resolved_code

# Accept input from stdin for VS Code plugin
if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        data = json.loads(raw_input)

        original = data.get("original", "")
        branch_a = data.get("branchA", "")
        branch_b = data.get("branchB", "")

        merged = resolve_merge_conflict(original, branch_a, branch_b)
        print(merged)
    except Exception as e:
        print(f"⚠️ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)