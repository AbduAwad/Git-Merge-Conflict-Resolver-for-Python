import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from dataPreprocessing import prepare_dataset, augment_dataset

# Define hyperparameters
MAX_LEN = 1024 
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
HIDDEN_SIZE = 256
EMBEDDING_DIM = 100  # Embedding size
NUM_CLASSES = 2  # Two classes (Branch A or B)
PAD_IDX = 0  # Padding index for embedding layer
UNK_IDX = 1  # Unknown word index

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class MergeConflictDatasetRNN(Dataset):
    def __init__(self, examples, word_to_idx, max_len):
        self.examples = examples
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def text_to_indices(self, text):
        """Convert text into a sequence of word indices."""
        return [self.word_to_idx.get(word, UNK_IDX) for word in text.split()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Convert text inputs into word index sequences
        original_indices = self.text_to_indices(example["original"])
        branch_a_indices = self.text_to_indices(example["branch_a"])
        branch_b_indices = self.text_to_indices(example["branch_b"])

        # Concatenate into a single input sequence
        input_sequence = original_indices + branch_a_indices + branch_b_indices

        # Pad sequence to max_len
        padded_input = input_sequence[:self.max_len] + [PAD_IDX] * (self.max_len - len(input_sequence))

        # Label: 0 if merged == branch_a, else 1
        label = 0 if example["merged"] == example["branch_a"] else 1

        return {
            "input_sequence": torch.tensor(padded_input, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Build Vocabulary
def build_vocab(examples):
    word_counter = Counter()
    for example in examples:
        for text in [example["original"], example["branch_a"], example["branch_b"]]:
            word_counter.update(text.split())

    # Assign an index to each word
    word_to_idx = {word: idx + 2 for idx, (word, _) in enumerate(word_counter.most_common())}
    word_to_idx["<PAD>"] = PAD_IDX  # Padding token
    word_to_idx["<UNK>"] = UNK_IDX  # Unknown words

    return word_to_idx

# Define RNN Classifier with BiLSTM
class MergeConflictRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(MergeConflictRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 because BiLSTM
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)  # Convert tokens into dense vectors
        lstm_out, _ = self.lstm(embedded)
        out = self.dropout(lstm_out[:, -1, :])  # Take last hidden state
        out = self.fc(out)
        return out

# Prepare Dataset
def prepare_rnn_dataset():
    # Load dataset using existing function
    train_examples, val_examples, test_examples = prepare_dataset()

    # Augment dataset if needed
    if len(train_examples) < 100:
        train_examples = augment_dataset(train_examples)

    # Build vocabulary
    word_to_idx = build_vocab(train_examples)

    # Create dataset objects
    train_dataset = MergeConflictDatasetRNN(train_examples, word_to_idx, MAX_LEN)
    val_dataset = MergeConflictDatasetRNN(val_examples, word_to_idx, MAX_LEN)
    test_dataset = MergeConflictDatasetRNN(test_examples, word_to_idx, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, word_to_idx

# Training Function
def train_rnn(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
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

        train_accuracy = correct_train / total_train

        # Validation Evaluation
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

        val_accuracy = correct_val / total_val
        model.train()  # Switch back to training mode

        # Print both Training and Validation Accuracies
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

# Evaluation Function
def evaluate_rnn(model, test_loader):
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader:
            input_sequence, labels = batch["input_sequence"].to(device), batch["label"].to(device)
            outputs = model(input_sequence)
            predictions = torch.argmax(outputs, dim=1)
            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)

    test_accuracy = correct_test / total_test
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Main Function to Train and Evaluate RNN Model
def main():
    train_loader, val_loader, test_loader, word_to_idx = prepare_rnn_dataset()

    # Initialize Model
    model = MergeConflictRNN(
        vocab_size=len(word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    ).to(device)

    # Train and Evaluate
    train_rnn(model, train_loader, val_loader)
    evaluate_rnn(model, test_loader)

if __name__ == "__main__":
    main()
