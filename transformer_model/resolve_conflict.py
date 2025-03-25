from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Specify the directory where you saved the model
model_path = "/content/drive/MyDrive/merge_conflict_model/final_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the trained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# If you also want to use the embedding model (SentenceTransformer)
embedding_model_name = "all-MiniLM-L6-v2" # Or the one you initialized with
embedding_model = SentenceTransformer(embedding_model_name)

# Now you can use 'model' and 'tokenizer' for inference, for example, in a
# modified 'resolve_merge_conflict' function or a new function.

# Example of using the loaded model for inference (simplified):
original_code = """
def greet(name):
    return "Hello, " + name
"""
branch_a_code = """
def greet(name):
    return f"Hi {name}!"
"""
branch_b_code = """
def greet(name):
    # Formal greeting
    return "Greetings, " + name
"""

input_text = (
    f"Resolve merge conflict between two code branches.\n"
    f"Original code:\n{original_code}\n\n"
    f"Branch A changes:\n{branch_a_code}\n\n"
    f"Branch B changes:\n{branch_b_code}\n\n"
    f"Provide the merged code:"
)

inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
resolved_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Resolved Code (from loaded model):")
print(resolved_code)