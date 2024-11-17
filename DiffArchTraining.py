import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from itertools import product
from typing import List, Dict, Tuple

# Fetch and prepare dataset
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


class ConcreteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], activation_fn):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, patience_plateau: int = 15):
        self.patience = patience
        self.patience_plateau = patience_plateau
        self.min_delta = min_delta
        self.counter = 0
        self.plateau_counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        self.stop_reason = None

    def __call__(self, val_loss: float, train_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_model(model)
            return False

        # Check for overfitting (validation loss increasing)
        if val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_reason = "overfitting"
                return True
        else:
            self.counter = 0
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(model)

        # Check for plateau in training loss
        if abs(train_loss - val_loss) < self.min_delta:
            self.plateau_counter += 1
            if self.plateau_counter >= self.patience_plateau:
                self.early_stop = True
                self.stop_reason = "plateau"
                return True
        else:
            self.plateau_counter = 0

        return False

    def save_model(self, model: nn.Module):
        self.best_model = model.state_dict().copy()


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_epochs: int = 1000
) -> Tuple[List[float], List[float], int, str]:
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping()

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if early_stopping(avg_val_loss, avg_train_loss, model):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs due to {early_stopping.stop_reason}")
            # Restore best model
            model.load_state_dict(early_stopping.best_model)
            return train_losses, val_losses, epoch + 1, early_stopping.stop_reason

        if (epoch + 1) % 20 == 0:
            print(f'\rEpoch [{epoch + 1}/{max_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}', end="")

    return train_losses, val_losses, max_epochs, "max_epochs_reached"


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[
    np.ndarray, np.ndarray, float, float]:
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
val_dataset = ConcreteDataset(X_val_scaled, y_val)
test_dataset = ConcreteDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define architectures and activation functions
architectures = [
    [64, 32],
    [128, 64, 32],
    [256, 128, 64, 32],
    [32, 32],
    [64]
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

for arch, (act_name, act_fn) in product(architectures, activation_fns.items()):
    model_name = f"MLP_{act_name}_{'_'.join(map(str, arch))}"
    print(f"\nTraining {model_name}")

    model = MLP(X_train.shape[1], arch, act_fn).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with early stopping
    train_losses, val_losses, epochs_run, stop_reason = train_model(
        model, train_loader, val_loader, criterion, optimizer, device
    )

    # Evaluate the model
    predictions, actuals, mse, r2 = evaluate_model(model, test_loader, device)

    results[model_name] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'mse': mse,
        'r2': r2,
        'epochs_run': epochs_run,
        'stop_reason': stop_reason
    }

# Plotting results
plt.figure(figsize=(20, 15))

# Plot training and validation losses
plt.subplot(2, 1, 1)
for model_name, result in results.items():
    plt.plot(result['train_losses'], label=f'{model_name} (train)')
    plt.plot(result['val_losses'], label=f'{model_name} (val)', linestyle='--')
plt.title('Training and Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Plot MSE comparison
plt.subplot(2, 1, 2)
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
    print(f"Epochs run: {result['epochs_run']}")
    print(f"Stopping reason: {result['stop_reason']}")