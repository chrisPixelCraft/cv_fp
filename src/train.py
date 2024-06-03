import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
import numpy as np
from model import CNNRNNModel
from dataset import DoorStateDatasetTrain
from utils import load_labels, pad_collate_fn

# data
frames_per_input = 5
spacing = 1

# model
model_dir = "./models"
input_size = 2048  # Example input size
rnn_hidden_size = 512
rnn_layers = 5
num_classes = 4

# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
learning_rate = 0.001
num_epochs = 100
batch_size = 16

# # Load labels
labels_path = '../data/labels/labels.json'
labels = load_labels(labels_path)

# Split into training and validation sets
train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

# Create datasets and data loaders
train_dataset = DoorStateDatasetTrain('../data/features', labels, train_idx, num_of_frames=frames_per_input, spacing=spacing)
val_dataset = DoorStateDatasetTrain('../data/features', labels, val_idx, num_of_frames=frames_per_input, spacing=spacing)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

# Initialize the model
model = CNNRNNModel(input_size=input_size, rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, rnn_layers=rnn_layers, dropout=0.5)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create directory for saving models if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# # Adjust learning rate if necessary
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Training loop

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)

            # outputs = outputs.view(-1, outputs.size(-1))  # Reshape for the loss function
            # labels = labels.view(-1)  # Reshape for the loss function

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            pbar.update(1)
            pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * train_loader.batch_size))

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)

            # outputs = outputs.view(-1, outputs.size(-1))  # Reshape for the loss function
            # labels = labels.view(-1)  # Reshape for the loss function

            loss = criterion(outputs, labels)
            val_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    # # Adjust learning rate
    # scheduler.step(val_loss)

    # Save the model checkpoint
    model_save_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')



