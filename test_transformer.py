# test the model at model_output_codet5_1000

from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
import torch

model_path = "/content/model_output_codet5_1000/final_model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

model.to("cuda" if torch.cuda.is_available() else "cpu")

original_code = "def add(a, b):\n    return a + b"
branch_a_code = "def add(a, b):\n    return a + b"
branch_b_code = "def add(a, b):\n    return a - b"

input_text = (
    f"<O>\n{original_code}\n</O>\n"
    f"<A>\n{branch_a_code}\n</A>\n"
    f"<B>\n{branch_b_code}\n</B>"
)
      

inputs = tokenizer(
    input_text,
    truncation=True,
    max_length=512,
    padding="max_length",
    return_tensors="pt"
)

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

