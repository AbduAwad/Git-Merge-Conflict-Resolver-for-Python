from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
import torch

# Load model and tokenizer
model_path = "/content/model_output_codet5_5000/final_model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define 5 test merge conflict examples
examples = [
    {
        "original": "def multiply(a, b):\n    return a * b",
        "a": "def multiply_numbers(a, b):\n    return a * b",
        "b": "def multiply(a, b):\n    result = a * b\n    return result"
    },
    {
        "original": "def greet(name):\n    print(f\"Hello, {name}!\")",
        "a": "def greet(name):\n    print(f\"Hi, {name}!\")",
        "b": "def greet(first_name, last_name):\n    print(f\"Hello, {first_name} {last_name}!\")"
    },
    {
        "original": "def divide(a, b):\n    return a / b",
        "a": "def divide(a, b):\n    print(\"Dividing numbers\")\n    return a / b",
        "b": "def divide(a, b):\n    return a / b if b != 0 else 0"
    },
    {
        "original": "def max_val(x, y):\n    return x if x > y else y",
        "a": "def max_val(x, y):\n    return max(x, y)",
        "b": "def max_val(x, y):\n    return x if x >= y else y"
    },
    {
        "original": "def is_even(n):\n    return n % 2 == 0",
        "a": "def is_even(n):\n    # check if number is even\n    return n % 2 == 0",
        "b": "def is_even(n):\n    return (n & 1) == 0"
    }
]

# Run inference on each example
for i, ex in enumerate(examples):
    input_text = (
        f"<O>\n{ex['original']}\n</O>\n"
        f"<A>\n{ex['a']}\n</A>\n"
        f"<B>\n{ex['b']}\n</B>"
    )
    
    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        max_length=768
    )

    resolved = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resolved = resolved.replace("<O>", "").replace("</O>", "") \
                       .replace("<A>", "").replace("</A>", "") \
                       .replace("<B>", "").replace("</B>", "")

    print(f"\n==== Example {i + 1} ====")
    print("Original Code:\n", ex["original"])
    print("Branch A:\n", ex["a"])
    print("Branch B:\n", ex["b"])
    print("Resolved Merge:\n", resolved)
