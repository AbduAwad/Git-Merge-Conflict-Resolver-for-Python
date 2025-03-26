import sys
import os
import torch
import json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataPreprocessing import prepare_dataset, process_file_content, augment_dataset

# Base paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(base_dir, "dataset", "conflicts-py")
synthetic_json_path = os.path.join(base_dir, "dataset", "synthetic_merge_conflicts_50000_batched.json")

# Hyperparameters
MAX_LEN = 1024 
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
HIDDEN_SIZE = 256
EMBEDDING_DIM = 100
NUM_CLASSES = 2
PAD_IDX = 0
UNK_IDX = 1

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load synthetic dataset
def load_synthetic_json_dataset(json_path, max_samples=1000):
    with open(json_path, 'r') as f:
        examples = json.load(f)

    if max_samples:
        examples = examples[:max_samples]

    train, test = train_test_split(examples, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    print(f"\ Loaded Synthetic Dataset: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test

# Dataset class
class MergeConflictDatasetRNN(Dataset):
    def __init__(self, examples, word_to_idx, max_len):
        self.examples = examples
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def text_to_indices(self, text):
        return [self.word_to_idx.get(word, UNK_IDX) for word in text.split()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        original_indices = self.text_to_indices(example["original"])
        branch_a_indices = self.text_to_indices(example["branch_a"])
        branch_b_indices = self.text_to_indices(example["branch_b"])

        input_sequence = original_indices + branch_a_indices + branch_b_indices
        padded_input = input_sequence[:MAX_LEN] + [PAD_IDX] * (MAX_LEN - len(input_sequence))

        sim_a = SequenceMatcher(None, example["merged"], example["branch_a"]).ratio()
        sim_b = SequenceMatcher(None, example["merged"], example["branch_b"]).ratio()
        label = 0 if sim_a > sim_b else 1

        return {
            "input_sequence": torch.tensor(padded_input, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Vocabulary
def build_vocab(examples):
    word_counter = Counter()
    for example in examples:
        for text in [example["original"], example["branch_a"], example["branch_b"]]:
            word_counter.update(text.split())

    word_to_idx = {word: idx + 2 for idx, (word, _) in enumerate(word_counter.most_common())}
    word_to_idx["<PAD>"] = PAD_IDX
    word_to_idx["<UNK>"] = UNK_IDX
    return word_to_idx

# PyTorch Model
class MergeConflictRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(MergeConflictRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(embedded)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

# Keras Model
def build_keras_model(vocab_size, embedding_dim, hidden_size, max_len, num_classes):
    inputs = Input(shape=(max_len,), dtype='int32', name='input_sequence')
   
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True)(inputs)
    x = Bidirectional(LSTM(hidden_size, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Dataset Preparation
def prepare_rnn_dataset():
    train_examples, val_examples, test_examples = load_synthetic_json_dataset(synthetic_json_path)
    word_to_idx = build_vocab(train_examples)

    train_dataset = MergeConflictDatasetRNN(train_examples, word_to_idx, MAX_LEN)
    val_dataset = MergeConflictDatasetRNN(val_examples, word_to_idx, MAX_LEN)
    test_dataset = MergeConflictDatasetRNN(test_examples, word_to_idx, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # For Keras Model
    def to_numpy(dataset):
        X, y = [], []
        for item in dataset:
            X.append(item['input_sequence'].numpy())
            y.append(item['label'].item())
        return np.array(X), np.array(y)

    keras_data = {
        'train': to_numpy(train_dataset),
        'val': to_numpy(val_dataset),
        'test': to_numpy(test_dataset)
    }

    return train_loader, val_loader, test_loader, word_to_idx, keras_data

# PyTorch Training
def train_rnn(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            input_sequence, labels = batch["input_sequence"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_sequence)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                input_sequence, labels = batch["input_sequence"].to(device), batch["label"].to(device)
                outputs = model(input_sequence)
                predictions = torch.argmax(outputs, dim=1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        model.train()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Evaluation
def evaluate_rnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_sequence, labels = batch["input_sequence"].to(device), batch["label"].to(device)
            outputs = model(input_sequence)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")

# Main entry point
if __name__ == "__main__":
    train_loader, val_loader, test_loader, word_to_idx, keras_data = prepare_rnn_dataset()

    print("\n===== PyTorch Training =====")
    model = MergeConflictRNN(
        vocab_size=len(word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    ).to(device)

    # train_rnn(model, train_loader, val_loader)
    # evaluate_rnn(model, test_loader)

    print("\n===== Keras Training =====")
    keras_model = build_keras_model(
        vocab_size=len(word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        max_len=MAX_LEN,
        num_classes=NUM_CLASSES
    )

    X_train, y_train = keras_data['train']
    X_val, y_val = keras_data['val']
    X_test, y_test = keras_data['test']

    keras_model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

    test_loss, test_acc = keras_model.evaluate(X_test, y_test)
    print(f"Keras Test Accuracy: {test_acc:.4f}")
