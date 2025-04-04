import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = '0'  # 🚫 Block TensorFlow/Keras imports in this file

import sys
import json
import torch
import ast
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# Path to trained model
model_path = os.path.join(os.path.dirname(__file__), "merge_conflict_model-20250325T222421Z-001", "merge_conflict_model", "final_model")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load embedding model for semantic evaluation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def advanced_preprocess(code):
    lines = [line for line in code.split('\n') if not line.startswith(('<<<<<<', '=======', '>>>>>>>'))]
    cleaned = [line.rstrip() for line in lines]
    imports, body = set(), []
    for line in cleaned:
        if line.startswith(('import ', 'from ')):
            imports.add(line)
        else:
            body.append(line)
    return '\n'.join(sorted(list(imports)) + body)

def is_valid_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_signature(code):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return f"{node.name}{ast.unparse(node.args)}"
    except:
        return ""
    return ""

def semantic_similarity(code1, code2):
    try:
        emb1 = embedding_model.encode([code1])[0]
        emb2 = embedding_model.encode([code2])[0]
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    except:
        return 0.0

def evaluate_merge_quality(original, merged):
    score = 0.0
    if is_valid_syntax(merged): score += 0.3
    if extract_signature(original) == extract_signature(merged): score += 0.2
    score += 0.3 * semantic_similarity(original, merged)
    line_diff = abs(len(original.split('\n')) - len(merged.split('\n')))
    score -= min(0.2, 0.05 * line_diff)
    return max(0, min(1.0, score))

def resolve_merge_conflict(original, branch_a, branch_b, max_tries=3):
    original = advanced_preprocess(original)
    branch_a = advanced_preprocess(branch_a)
    branch_b = advanced_preprocess(branch_b)

    input_text = (
        f"Resolve merge conflict between two code branches.\n"
        f"Original code:\n{original}\n\n"
        f"Branch A changes:\n{branch_a}\n\n"
        f"Branch B changes:\n{branch_b}\n\n"
        f"Provide the merged code:"
    )

    candidates = []

    for attempt in range(max_tries):
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7 + 0.1 * attempt,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded = advanced_preprocess(decoded)
        if is_valid_syntax(decoded):
            score = evaluate_merge_quality(original, decoded)
            candidates.append((decoded, score))

    if candidates:
        best = max(candidates, key=lambda x: x[1])[0]
        return best
    else:
        return "// No confident merge result, returning original:\n" + original

if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        data = json.loads(raw_input)
        original = data.get("original", "")
        branch_a = data.get("branchA", "")
        branch_b = data.get("branchB", "")
        result = resolve_merge_conflict(original, branch_a, branch_b)
        print(result)
    except Exception as e:
        print(f"⚠️ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)