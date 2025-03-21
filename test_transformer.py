# Run in Google Colab for: (We Will Use this File in the plugin, when the time comes).

from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Specify the path to your saved model folder
model_path = "./model_output_2/final_model"

# Load the tokenizer and model from the folder
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare a new merge conflict example
original_code = """
def add(a, b):
    return a + b
"""

branch_a_code = """
def add(a, b):
    return a + b + 1
"""

branch_b_code = """
def add(a, b):
    return a + b - 1
"""

input_text = (
    f"<O>\n{original_code}\n</O>\n"
    f"<A>\n{branch_a_code}\n</A>\n"
    f"<B>\n{branch_b_code}\n</B>"
)


# Tokenize the input
inputs = tokenizer(
    input_text, 
    return_tensors="pt", 
    padding="max_length", 
    truncation=True, 
    max_length=768
)

# Move tensors to the correct device
inputs = {key: val.to("cuda" if torch.cuda.is_available() else "cpu") for key, val in inputs.items()}

# Generate a merge resolution using beam search
outputs = model.generate(
    **inputs, 
    num_beams=4, 
    early_stopping=True, 
    no_repeat_ngram_size=2, 
    max_length=768
)

# Decode the generated merge resolution
resolved_merge = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Resolved Merge:\n", resolved_merge)
