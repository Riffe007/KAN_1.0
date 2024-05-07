# File: main.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.kan_former import KANFormer
from utils.rotation_matrix import get_rotation_matrix

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    # Hyperparameters
    config = {
        'vocabulary_size': 10000,
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'max_length': 100,
        'num_experts': 10,
        'n_experts_per_token': 2,
        'd_ff': 2048,
    }
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup
    rotation_matrix = get_rotation_matrix(config['hidden_size'], config['max_length'], 10000.0)
    model = KANFormer(config).to(device)

    # Data setup (dummy data for illustration)
    dummy_data = torch.randint(0, config['vocabulary_size'], (1000, config['max_length']))
    dummy_targets = torch.randint(0, config['vocabulary_size'], (1000, config['max_length']))
    dataset = TensorDataset(dummy_data, dummy_targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, data_loader, optimizer, criterion, device)
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'kan_former_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
