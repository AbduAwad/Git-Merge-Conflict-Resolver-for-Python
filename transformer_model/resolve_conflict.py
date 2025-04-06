import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = '0'  # 🚫 Block TensorFlow/Keras imports in this file

# We suppress TensorFlow logs and block its usage entirely in this script to prevent unnecessary imports and conflicts."
# We rely only on PyTorch for inference here."

# We import essential packages for inference, JSON I/O, AST parsing for syntax validation, and our pre-trained transformer models."
import sys
import json
import torch
import ast
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# "This specifies the path to our previously saved fine-tuned model."
model_path = os.path.join(os.path.dirname(__file__), "merge_conflict_model-20250325T222421Z-001", "merge_conflict_model", "final_model")

# We load the tokenizer and model onto the available device (GPU or CPU). These handle the sequence-to-sequence merge generation task."
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# We also load a sentence transformer model for computing semantic similarity — used for scoring merge candidates."
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 

### HELPER FUNCTIONS FOR INFERENCE: 

# Cleans up merge markers like <<<<<<< and deduplicates import statements. Prepares the code for cleaner input to the model."
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

# Checks if the code has valid Python syntax using AST parsing. Returns True if valid, False otherwise."
def is_valid_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# Extracts the function signature from the code using AST parsing. Returns the function name and arguments."
def extract_signature(code):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return f"{node.name}{ast.unparse(node.args)}"
    except:
        return ""
    return ""

# Computes semantic similarity between two pieces of code using the sentence transformer model. Returns a float score."
def semantic_similarity(code1, code2):
    try:
        emb1 = embedding_model.encode([code1])[0]
        emb2 = embedding_model.encode([code2])[0]
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    except:
        return 0.0

# Evaluates the quality of the merged code based on syntax validity, signature match, semantic similarity, and line difference."
def evaluate_merge_quality(original, merged):
    score = 0.0
    if is_valid_syntax(merged): score += 0.3
    if extract_signature(original) == extract_signature(merged): score += 0.2
    score += 0.3 * semantic_similarity(original, merged)
    line_diff = abs(len(original.split('\n')) - len(merged.split('\n')))
    score -= min(0.2, 0.05 * line_diff)
    return max(0, min(1.0, score))

# This function resolves merge conflicts by generating candidate merges and selecting the best one based on quality evaluation."
def resolve_merge_conflict(original, branch_a, branch_b, max_tries=3):

    # We clean up all three versions using advanced_preprocess, which:
    original = advanced_preprocess(original)
    branch_a = advanced_preprocess(branch_a)
    branch_b = advanced_preprocess(branch_b)

    input_text = ( # This is the input text for the model, We build a natural language prompt to feed into the transformer.
        f"Resolve merge conflict between two code branches.\n"
        f"Original code:\n{original}\n\n"
        f"Branch A changes:\n{branch_a}\n\n"
        f"Branch B changes:\n{branch_b}\n\n"
        f"Provide the merged code:"
    )

    candidates = []

    # We’ll generate up to max_tries candidate outputs. This is helpful because generation is probabilistic — multiple attempts let us explore variations.
    for attempt in range(max_tries):
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device) # pass in the input text
        outputs = model.generate(  # use the model to generate a response
            **inputs, # pass in the inputs
            max_length=512, # max length of the output
            num_return_sequences=1, # number of sequences to return
            do_sample=True, # use sampling to allows for temprature, topk, and top_p
            temperature=0.7 + 0.1 * attempt, # The temperature is gradually increased per attempt, encouraging more exploration on subsequent tries.
            top_k=50, # top-k sampling, which limits the sampling pool to the top k most probable tokens.
            top_p=0.95, # nucleus sampling, which limits the sampling pool to the top p cumulative probability.
            repetition_penalty=1.2 # This discourages the model from repeating the same tokens in the output.
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True) # decoder to convert the output to text
        decoded = advanced_preprocess(decoded) # clean up the decoded output 
        if is_valid_syntax(decoded): # check if the decoded output is valid syntax
            score = evaluate_merge_quality(original, decoded) # evaluate the quality of the decoded output
            candidates.append((decoded, score)) # add the decoded output and score to the candidates list

    if candidates: # If we have candidates, we sort them by score and select the best one.
        best = max(candidates, key=lambda x: x[1])[0] # get the best candidate
        return best 
    else:
        return "// No confident merge result, returning original:\n" + original

if __name__ == "__main__": # This main function is called by the plugin and it recieves the input from stdin.
    # We read the input JSON from stdin, which contains the original code and two branches.
    try:
        raw_input = sys.stdin.read()
        data = json.loads(raw_input)
        original = data.get("original", "")
        branch_a = data.get("branchA", "")
        branch_b = data.get("branchB", "")
        result = resolve_merge_conflict(original, branch_a, branch_b) # run the merge conflict resolution
        print(result) # print the result to stdout (which is accessed by the plugin)
    except Exception as e:
        print(f"⚠️ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)