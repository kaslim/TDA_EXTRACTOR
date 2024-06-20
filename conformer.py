import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_blocks, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        for block in self.encoder_blocks:
            x = block(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * signals.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in dataloader:
            outputs = model(signals)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# Example usage
if __name__ == "__main__":
    # Simulated data for example purposes
    signals = np.random.randn(100, 250, 1)  # (num_samples, seq_len, input_dim)
    labels = np.random.randint(0, 2, size=100)  # Binary labels

    # Create dataset and dataloader
    dataset = ECGDataset(signals, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model parameters
    input_dim = 1
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_blocks = 2
    num_classes = 1
    dropout = 0.1
    num_epochs = 20

    # Initialize model, loss function, and optimizer
    model = TransformerClassifier(input_dim, embed_dim, num_heads, ff_dim, num_blocks, num_classes, dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)
    evaluate_model(model, dataloader)
