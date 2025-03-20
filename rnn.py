import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from transformers import AutoTokenizer
from dataPreprocessing import *

# Set device (CPU-based training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained tokenizer (e.g., CodeBERT or BART)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Hyperparameters
MAX_LEN = 512  # Max sequence length
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
HIDDEN_SIZE = 256
NUM_CLASSES = 2  # Two classes (Branch A or B)
EMBEDDING_DIM = 768  # CodeBERT embedding size

# Custom Dataset Class
class MergeConflictDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Construct input as "<O> original code <A> branch A <B> branch B"
        input_text = f"<O>\n{example['original']}\n</O>\n<A>\n{example['branch_a']}\n</A>\n<B>\n{example['branch_b']}\n</B>"
        label = 0 if example["merged"] == example["branch_a"] else 1  # 0: Select A, 1: Select B

        # Tokenization
        encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Define RNN Classifier with BiLSTM
class MergeConflictRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes):
        super(MergeConflictRNN, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take last output for classification
        out = self.fc(out)
        return out

# Prepare Dataset
def prepare_rnn_dataset():
    train_examples, val_examples, test_examples = prepare_dataset()
    train_dataset = MergeConflictDataset(train_examples, tokenizer, MAX_LEN)
    val_dataset = MergeConflictDataset(val_examples, tokenizer, MAX_LEN)
    test_dataset = MergeConflictDataset(test_examples, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

# Training Function
def train_rnn(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            optimizer.zero_grad()

            # Convert input to embeddings
            with torch.no_grad():
                embeddings = tokenizer(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluation Function
def evaluate_rnn(model, test_loader):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            # Convert input to embeddings
            embeddings = tokenizer(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

            outputs = model(embeddings)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(pred)
            actuals.extend(labels)

    # Compute Evaluation Metrics
    acc = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def main():
    train_loader, val_loader, test_loader = prepare_rnn_dataset()

    # Initialize Model
    model = MergeConflictRNN(EMBEDDING_DIM, HIDDEN_SIZE, NUM_CLASSES).to(device)

    # Train and Evaluate
    train_rnn(model, train_loader, val_loader)
    evaluate_rnn(model, test_loader)

#if __name__ == "__main__":
#   main()
