"""
The VsCode Extension calls this script to resolve the merge conflict.
"""

from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
import torch
import re
import sys
import json
import os
import time

# Load model and tokenizer
model_path = r"C:\Users\sheri\OneDrive - Carleton University\Comp_Courses\Comp4107\FINAL_PROJECT\COMP4107_FP\transformer_model\model_output_codet5\final_model"

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def resolve_merge_conflict(original, a, b):
    input_text = (
        f"Resolve the following merge conflict, ensuring the output code is correct:\n"
        f"<ORIGINAL>\n{original}\n</ORIGINAL>\n"
        f"<BRANCH_A>\n{a}\n</BRANCH_A>\n"
        f"<BRANCH_B>\n{b}\n</BRANCH_B>\n"
        f"<MERGED>"
    )


    # ✅ Enhanced tokenization with more robust handling
    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=752,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ✅ More sophisticated generation parameters
    outputs = model.generate(
        **inputs,
        max_length=752,
        num_beams=8,  # Increased beam search
        early_stopping=True,
        do_sample=True,
        no_repeat_ngram_size=3,  # Prevent repetitive generations
        temperature=0.7,  # Add some controlled randomness
        top_k=50,  # Top-k sampling for diversity
        top_p=0.95,  # Nucleus sampling
        repetition_penalty=1.2,  # Slightly discourage repeating phrases
    )

    # ✅ More robust decoding
    merged = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Advanced cleanup and formatting
    merged = re.sub(r'\n{3,}', '\n\n', merged).strip()
    return merged if merged else None

# 🔄 Read input from stdin (from the extension)
if __name__ == "__main__":
    input_json = sys.stdin.read()
    if not input_json.strip():
        print("⚠️ No input received.")
        sys.exit(1)

    try:
        payload = json.loads(input_json)
        result = resolve_merge_conflict(payload['original'], payload['branchA'], payload['branchB'])
        print(result)
    except Exception as e:
        print(f"⚠️ Error resolving conflict: {e}", file=sys.stderr)
        sys.exit(1)
