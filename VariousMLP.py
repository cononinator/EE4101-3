import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from itertools import product

# Fetch and prepare dataset
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Dataset class
class ConcreteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Modified MLP class to support different architectures
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_fn):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Training function
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    return train_losses


# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions).reshape(-1)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return predictions, actuals, mse, r2


# Create data loaders
train_dataset = ConcreteDataset(X_train_scaled, y_train)
test_dataset = ConcreteDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define different architectures and activation functions to try
architectures = [
    [64, 32],  # Original architecture
    [128, 64, 32],  # Deeper architecture
    [256, 128, 64, 32],  # Even deeper architecture
    [32, 32],  # Symmetric architecture
    [64]  # Single hidden layer
]

activation_fns = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'GELU': nn.GELU,
    'Tanh': nn.Tanh
}

# Dictionary to store results
results = {}

# Train and evaluate different models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100

for arch, (act_name, act_fn) in product(architectures, activation_fns.items()):
    model_name = f"MLP_{act_name}_{'_'.join(map(str, arch))}"
    print(f"\nTraining {model_name}")

    model = MLP(X_train.shape[1], arch, act_fn).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs, device)

    # Evaluate the model
    predictions, actuals, mse, r2 = evaluate_model(model, test_loader, device)

    results[model_name] = {
        'train_losses': train_losses,
        'predictions': predictions,
        'mse': mse,
        'r2': r2
    }

# Plotting results
plt.figure(figsize=(20, 10))

# Plot training losses
plt.subplot(1, 2, 1)
for model_name, result in results.items():
    plt.plot(result['train_losses'], label=model_name)
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Plot MSE comparison
plt.subplot(1, 2, 2)
model_names = list(results.keys())
mse_values = [results[model]['mse'] for model in model_names]
plt.bar(range(len(model_names)), mse_values)
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.title('MSE Comparison')
plt.ylabel('Mean Squared Error')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Results:")
print("-" * 50)
for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"MSE: {result['mse']:.2f}")
    print(f"R² Score: {result['r2']:.2f}")