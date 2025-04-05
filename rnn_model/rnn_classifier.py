# This file is called by the plugin to resolve a merge conflict by using the trained RNN model.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF info/warning/debug logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs

import json
import sys
import numpy as np
import tensorflow as tf

# Constants
MAX_LEN = 1024
PAD_IDX = 0
UNK_IDX = 1

# Load vocab
def load_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        return {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    with open(vocab_path, 'r') as f:
        return json.load(f)

# Convert text to padded index tensor
def text_to_indices(text, word_to_idx):
    tokens = text.split()
    indices = [word_to_idx.get(token, UNK_IDX) for token in tokens]
    if len(indices) > MAX_LEN:
        indices = indices[:MAX_LEN]
    else:
        indices += [PAD_IDX] * (MAX_LEN - len(indices))
    return np.array(indices, dtype=np.int32)

def main():
    try:
        input_data = json.loads(sys.stdin.read())
        branch_a = input_data.get("branchA", "")
        branch_b = input_data.get("branchB", "")
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input", "selected_branch": "A", "confidence": 0.5}))
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "rnn_model.keras")
    vocab_path = os.path.join(script_dir, "rnn_vocab.json")

    try:
        word_to_idx = load_vocab(vocab_path)

        # Load TensorFlow model
        model = tf.keras.models.load_model(model_path)

        # Process inputs
        a_indices = text_to_indices(branch_a, word_to_idx)
        b_indices = text_to_indices(branch_b, word_to_idx)

        # Concatenate inputs and truncate to MAX_LEN
        input_indices = np.concatenate([a_indices, b_indices])
        input_indices = input_indices[:MAX_LEN]
        input_tensor = np.expand_dims(input_indices, axis=0)

        # Predict
        prediction = model.predict(input_tensor, verbose=0)
        if prediction.shape[1] > 1:
            prediction_class = int(np.argmax(prediction))
            confidence = float(prediction[0][prediction_class])
        else:
            prob = float(prediction[0][0])
            prediction_class = 1 if prob > 0.5 else 0
            confidence = prob if prediction_class == 1 else 1 - prob

        result = {
            "selected_branch": "A" if prediction_class == 0 else "B",
            "confidence": confidence,
            "resolved_code": branch_a if prediction_class == 0 else branch_b
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "selected_branch": "A", "confidence": 0.5, "resolved_code": branch_a}))

if __name__ == "__main__":
    main()
    # This is the entry point for the script when executed directly