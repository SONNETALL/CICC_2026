import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MNISTCSVDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        data = df.values
        self.y = data[:, 0].astype(np.int64)
        self.x = data[:, 1:].astype(np.float32) / 255.0  # 归一化到 [0,1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    train_csv = "./data/mnist_train.csv"
    test_csv = "./data/mnist_test.csv"
    save_path = "./ckpt/mnist_mlp_fp32.pth"

    os.makedirs("./ckpt", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = MNISTCSVDataset(train_csv)
    test_set = MNISTCSVDataset(test_csv)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    model = MLP(in_dim=784, hidden_dim=256, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] loss={total_loss:.4f} test_acc={acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
