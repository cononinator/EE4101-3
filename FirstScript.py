import numpy as np
import pandas as pd
import torch_directml
from numpy.ma.core import indices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scienceplots
from ucimlrepo import fetch_ucirepo
import psutil
import time

plt.style.use(['science', 'ieee'])

def print_gpu_utilization():
    # DirectML doesn't have direct GPU memory queries, so we'll monitor CPU memory transfer
    print(f"CPU Memory used: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")


class ConcreteDataset(Dataset):
    def __init__(self, X, y, device):
        # Move data to GPU during initialization
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize device
device = torch.device("cpu")
print(f"Using device: {device}")
print_gpu_utilization()

# Create data loaders with larger batch size
batch_size = 128  # Increased batch size for better GPU utilization
train_dataset = ConcreteDataset(X_train_scaled, y_train, device)
test_dataset = ConcreteDataset(X_test_scaled, y_test, device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False)

# Initialize model, loss function, and optimizer
mlp_model = MLP(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mlp_model.parameters(), lr=0.001)

# Training loop with test loss tracking
epochs = 100
train_losses = []
test_losses = []
print("Starting training...")
print_gpu_utilization()

start_time = time.time()

for epoch in range(epochs):
    # Training phase
    mlp_model.train()
    epoch_loss = 0
    batch_times = []

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        batch_start = time.time()

        # Data is already on GPU from the Dataset class
        optimizer.zero_grad()
        outputs = mlp_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_times.append(time.time() - batch_start)

    train_losses.append(epoch_loss / len(train_loader))

    # Evaluation phase for test loss calculation
    mlp_model.eval()
    test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = mlp_model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

    test_losses.append(test_loss / len(test_loader))

    # Print metrics every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
        print(f'Average batch time: {np.mean(batch_times):.4f} seconds')
        print_gpu_utilization()

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")

# Evaluation
mlp_model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = mlp_model(X_batch)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

predictions = np.array(predictions).reshape(-1)
actuals = np.array(actuals)

mlp_mse = mean_squared_error(actuals, predictions)
mlp_r2 = r2_score(actuals, predictions)

print("\nMLP Results:")
print(f"MSE: {mlp_mse:.2f}")
print(f"R2 Score: {mlp_r2:.2f}")

# Plotting results
plt.figure()

# Training and Test loss plot
# plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('MLP Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

plt.figure()
# Prediction comparison plot
# plt.subplot(1, 2, 2)
indices = np.random.choice(len(actuals), size=50, replace=False)
sampled_actuals = actuals[indices]
sampled_predictions = predictions[indices]
plt.scatter(actuals, predictions, alpha=0.5, label='MLP')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r')
plt.xlabel('Actual Values (GPa)')
plt.ylabel('Predicted Values (GPa)')
plt.title('Predictions vs Actuals')
plt.legend()

plt.tight_layout()
plt.show()