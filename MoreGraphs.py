import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from itertools import product
from collections import defaultdict
import time

plt.style.use(['science', 'ieee'])

# Fetch dataset
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets


class ConcreteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_fn):
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
    def __init__(self, patience=30, min_delta=0.05, patience_plateau=15,
                 plateau_window=10, max_oscillations=5, oscillation_threshold=0.02):
        """
        Args:
            patience (int): How many epochs to wait after validation loss increase before stopping
            min_delta (float): Minimum absolute change in validation loss to qualify as an improvement
            patience_plateau (int): How many epochs to wait when training plateaus
            plateau_window (int): Number of epochs to look back for plateau detection
            max_oscillations (int): Maximum number of oscillations before considering plateau
            oscillation_threshold (float): Threshold for considering a change as an oscillation
        """
        self.patience = patience
        self.min_delta = min_delta
        self.patience_plateau = patience_plateau
        self.plateau_window = plateau_window
        self.max_oscillations = max_oscillations
        self.oscillation_threshold = oscillation_threshold

        self.counter = 0
        self.plateau_counter = 0
        self.best_loss = None
        self.loss_history = []
        self.early_stop = False
        self.best_model = None
        self.stop_reason = None

    def count_oscillations(self):
        """Count the number of oscillations in the recent loss history."""
        if len(self.loss_history) < 3:
            return 0

        # Get differences between consecutive losses
        diffs = np.diff(self.loss_history[-self.plateau_window:])

        # Count sign changes that are significant (beyond threshold)
        oscillations = 0
        for i in range(len(diffs) - 1):
            if (abs(diffs[i]) > self.oscillation_threshold and
                    abs(diffs[i + 1]) > self.oscillation_threshold and
                    diffs[i] * diffs[i + 1] < 0):  # Sign change
                oscillations += 1

        return oscillations

    def check_plateau(self):
        """Check if the loss has plateaued by analyzing oscillations and trend."""
        if len(self.loss_history) < self.plateau_window:
            return False

        recent_losses = np.array(self.loss_history[-self.plateau_window:])

        # Check for oscillations
        num_oscillations = self.count_oscillations()

        # Check for overall improvement
        start_loss = np.mean(recent_losses[:3])  # Average first 3 in window
        end_loss = np.mean(recent_losses[-3:])  # Average last 3 in window
        relative_improvement = abs((start_loss - end_loss) / start_loss) if start_loss != 0 else 0

        # Calculate statistics to check for stable oscillation
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        coefficient_of_variation = loss_std / loss_mean if loss_mean != 0 else 0

        # Consider it a plateau if we see:
        # 1. Many oscillations with small overall improvement, or
        # 2. Small variation around mean with enough samples
        return ((num_oscillations >= self.max_oscillations and relative_improvement < self.oscillation_threshold) or
                (coefficient_of_variation < self.oscillation_threshold and len(recent_losses) >= self.plateau_window))

    def __call__(self, val_loss, train_loss, model):
        self.loss_history.append(val_loss)

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_model(model)
            return False

        # More permissive overfitting check
        if val_loss > (self.best_loss + self.min_delta):
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

        # Check for plateau
        if self.check_plateau():
            self.plateau_counter += 1
            if self.plateau_counter >= self.patience_plateau:
                self.early_stop = True
                self.stop_reason = "plateau"
                return True
        else:
            self.plateau_counter = 0

        return False

    def save_model(self, model):
        self.best_model = model.state_dict().copy()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=1000, verbose=False):
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
            if verbose:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs due to {early_stopping.stop_reason}")
            model.load_state_dict(early_stopping.best_model)
            return train_losses, val_losses, epoch + 1, early_stopping.stop_reason

        if verbose and (epoch + 1) % 20 == 0:
            print(f'\rEpoch [{epoch + 1}/{max_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}',
                  end='')

    return train_losses, val_losses, max_epochs, "max_epochs_reached"


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

    metrics = {
        'mse': mean_squared_error(actuals, predictions),
        'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
        'mae': mean_absolute_error(actuals, predictions),
        'mape': mean_absolute_percentage_error(actuals, predictions),
        'r2': r2_score(actuals, predictions)
    }

    return predictions, actuals, metrics


def plot_training_curves(results, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(results['train_losses'], label='Training Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.title(
        f'Training Curves - {model_name}\nStopped after {results["epochs_run"]} epochs ({results["stop_reason"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_comparison(final_results):
    metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 6))

    for i, metric in enumerate(metrics):
        data = [(model, results[metric]['mean'], results[metric]['std'])
                for model, results in final_results.items()]
        models, means, stds = zip(*sorted(data, key=lambda x: x[1]))

        axes[i].barh(range(len(models)), means, xerr=stds, height=0.8)
        axes[i].set_yticks(range(len(models)))
        axes[i].set_yticklabels([m.split('_', 1)[1] for m in models], fontsize=8)
        axes[i].set_title(f'{metric.upper()}')
        axes[i].grid(True)

        # Rotate long labels
        if i == 0:  # Only for the first subplot
            axes[i].tick_params(axis='y', labelrotation=0)

    plt.tight_layout()
    plt.show()


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

# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionary to store results
cv_results = defaultdict(lambda: defaultdict(list))
training_histories = defaultdict(list)
all_predictions = defaultdict(list)

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_splits}")
    print("-" * 80)
    startTime = time.time()

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create data loaders
    train_dataset = ConcreteDataset(X_train_scaled, y_train)
    val_dataset = ConcreteDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for arch, (act_name, act_fn) in product(architectures, activation_fns.items()):
        model_name = f"MLP_{act_name}_{'_'.join(map(str, arch))}"
        print(f"\nTraining {model_name}")

        model = MLP(X_train.shape[1], arch, act_fn).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train with early stopping
        train_losses, val_losses, epochs_run, stop_reason = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, verbose=True
        )

        # Evaluate
        predictions, actuals, metrics = evaluate_model(model, val_loader, device)

        # Store results
        for metric_name, value in metrics.items():
            cv_results[model_name][metric_name].append(value)

        training_histories[model_name].append({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_run': epochs_run,
            'stop_reason': stop_reason
        })

        all_predictions[model_name].extend(list(zip(actuals, predictions)))
    print(f"Fold {fold + 1} time taken: {time.time() - startTime}")

# Calculate final results
final_results = {}
for model_name in cv_results:
    final_results[model_name] = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in cv_results[model_name].items()
    }

# Print detailed results
print("\nDetailed Cross-Validation Results:")
print("-" * 80)
for model_name, metrics in final_results.items():
    print(f"\n{model_name}:")
    avg_epochs = np.mean([h['epochs_run'] for h in training_histories[model_name]])
    stop_reasons = [h['stop_reason'] for h in training_histories[model_name]]
    print(f"Average epochs: {avg_epochs:.1f}")
    print(f"Stop reasons: {dict(pd.Series(stop_reasons).value_counts())}")
    for metric, values in metrics.items():
        print(f"{metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")

# Generate plots
plot_metrics_comparison(final_results)

# Plot training curves for best model
best_model = min(final_results.items(), key=lambda x: x[1]['mse']['mean'])[0]
print(f"\nPlotting training curves for best model: {best_model}")
for fold, history in enumerate(training_histories[best_model]):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title(
        f'{best_model} - Fold {fold + 1}\nStopped after {history["epochs_run"]} epochs ({history["stop_reason"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()