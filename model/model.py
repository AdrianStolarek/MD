import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

class DummyDataset(Dataset):
    def __init__(self, size=1000, dim=10):
        self.size = size
        self.dim = dim
        self.X = torch.randn(size, dim)
        self.y = torch.randint(0, 2, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, epochs=5, lr=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%')
    
    return model

def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test accuracy: {accuracy:.2f}%')
    
    metrics = {
        "accuracy": accuracy
    }
    
    return metrics

def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model zapisany w {path}")
    
def save_metrics(metrics, path="metrics.json"):
    with open(path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metryki zapisane w {path}")

def main():
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = os.environ.get("AIP_MODEL_DIR", ".")
    
    train_dataset = DummyDataset(size=1000, dim=10)
    test_dataset = DummyDataset(size=200, dim=10)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model = SimpleModel(input_dim=10)
    
    trained_model = train_model(
        model, 
        train_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        device=DEVICE
    )
    
    metrics = evaluate_model(trained_model, test_loader, device=DEVICE)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(trained_model, os.path.join(MODEL_DIR, "model.pt"))
    save_metrics(metrics, os.path.join(MODEL_DIR, "metrics.json"))
    
if __name__ == "__main__":
    main()