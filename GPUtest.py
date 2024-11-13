import numpy as np
import pandas as pd
import torch_directml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import psutil
import time

def print_memory_utilization():
    # DirectML doesn't have direct GPU memory queries, so we'll monitor CPU memory transfer
    print(f"CPU Memory used: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

class ConcreteDataset(Dataset):
    def __init__(self, X, y, device):
        # Move data to specified device during initialization
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y.values).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),  # Increased layer size
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Added BatchNorm
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Added BatchNorm
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Fetch and prepare data
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize device for GPU (DirectML)
device_gpu = torch_directml.device()
print(f"Using GPU device: {device_gpu}")
print_memory_utilization()

# Create data loaders for GPU with larger batch size
batch_size = 256
train_dataset_gpu = ConcreteDataset(X_train_scaled, y_train, device_gpu)
test_dataset_gpu = ConcreteDataset(X_test_scaled, y_test, device_gpu)
train_loader_gpu = DataLoader(train_dataset_gpu, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader_gpu = DataLoader(test_dataset_gpu, batch_size=batch_size, pin_memory=False)

# Initialize model, loss function, and optimizer for GPU
mlp_model_gpu = MLP(X_train.shape[1]).to(device_gpu)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mlp_model_gpu.parameters(), lr=0.001)

# Training loop for GPU
epochs = 100
train_losses = []
print("Starting GPU training...")
print_memory_utilization()

start_time = time.time()

for epoch in range(epochs):
    mlp_model_gpu.train()
    epoch_loss = 0
    batch_times = []

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader_gpu):
        batch_start = time.time()

        # Data is already on GPU from the Dataset class
        optimizer.zero_grad()
        outputs = mlp_model_gpu(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_times.append(time.time() - batch_start)

    train_losses.append(epoch_loss / len(train_loader_gpu))

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_losses[-1]:.4f}')
        print(f'Average batch time: {np.mean(batch_times):.4f} seconds')
        print_memory_utilization()

print(f"\nGPU training completed in {time.time() - start_time:.2f} seconds")

# Evaluation on GPU
mlp_model_gpu.eval()
predictions_gpu = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader_gpu:
        outputs = mlp_model_gpu(X_batch)
        predictions_gpu.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

predictions_gpu = np.array(predictions_gpu).reshape(-1)
actuals = np.array(actuals)

mlp_mse_gpu = mean_squared_error(actuals, predictions_gpu)
mlp_r2_gpu = r2_score(actuals, predictions_gpu)

print("\nGPU Results:")
print(f"MSE: {mlp_mse_gpu:.2f}")
print(f"R2 Score: {mlp_r2_gpu:.2f}")

# Run on CPU
device_cpu = torch.device("cpu")
train_dataset_cpu = ConcreteDataset(X_train_scaled, y_train, device_cpu)
test_dataset_cpu = ConcreteDataset(X_test_scaled, y_test, device_cpu)
train_loader_cpu = DataLoader(train_dataset_cpu, batch_size=batch_size, shuffle=True)
test_loader_cpu = DataLoader(test_dataset_cpu, batch_size=batch_size)

mlp_model_cpu = MLP(X_train.shape[1]).to(device_cpu)
optimizer_cpu = torch.optim.SGD(mlp_model_cpu.parameters(), lr=0.001)

print("\nStarting CPU training...")
start_time = time.time()

for epoch in range(epochs):
    mlp_model_cpu.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader_cpu:
        optimizer_cpu.zero_grad()
        outputs = mlp_model_cpu(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_cpu.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader_cpu):.4f}')

print(f"\nCPU training completed in {time.time() - start_time:.2f} seconds")

# Evaluation on CPU
mlp_model_cpu.eval()
predictions_cpu = []

with torch.no_grad():
    for X_batch, y_batch in test_loader_cpu:
        outputs = mlp_model_cpu(X_batch)
        predictions_cpu.extend(outputs.numpy())

predictions_cpu = np.array(predictions_cpu).reshape(-1)

mlp_mse_cpu = mean_squared_error(actuals, predictions_cpu)
mlp_r2_cpu = r2_score(actuals, predictions_cpu)

print("\nCPU Results:")
print(f"MSE: {mlp_mse_cpu:.2f}")
print(f"R2 Score: {mlp_r2_cpu:.2f}")

# Plotting results
plt.figure(figsize=(12, 5))

# Training loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('MLP Training Loss (GPU)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

# Prediction comparison plot
plt.subplot(1, 2, 2)
plt.scatter(actuals, predictions_gpu, alpha=0.5, label='GPU Predictions')
plt.scatter(actuals, predictions_cpu, alpha=0.5, label='CPU Predictions')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actuals')
plt.legend()

plt.tight_layout()
plt.show()
