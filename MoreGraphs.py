
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from itertools import product, count
from collections import defaultdict

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


def train_model(model, train_loader, criterion, optimizer, epochs, device, verbose=False):
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

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    return train_losses


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
epochs = 100

# Dictionary to store results
cv_results = defaultdict(lambda: defaultdict(list))
all_predictions = defaultdict(list)

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_splits}")

    # Split and scale data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create data loaders
    train_dataset = ConcreteDataset(X_train_scaled, y_train)
    val_dataset = ConcreteDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    count = 0

    for arch, (act_name, act_fn) in product(architectures, activation_fns.items()):
        model_name = f"MLP_{act_name}_{'_'.join(map(str, arch))}"
        print(f"\rCount: {count} Training: {model_name}", end="", flush=True)

        model = MLP(X_train.shape[1], arch, act_fn).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train and evaluate
        train_losses = train_model(model, train_loader, criterion, optimizer, epochs, device)
        predictions, actuals, metrics = evaluate_model(model, val_loader, device)

        # Store results
        for metric_name, value in metrics.items():
            cv_results[model_name][metric_name].append(value)
        all_predictions[model_name].extend(list(zip(actuals, predictions)))
        count += 1

# Calculate mean and std of metrics across folds
final_results = {}
for model_name in cv_results:
    final_results[model_name] = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in cv_results[model_name].items()
    }


# Plotting functions
def plot_metrics_comparison(final_results):
    metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5 * n_metrics))

    for i, metric in enumerate(metrics):
        means = [final_results[model][metric]['mean'] for model in final_results]
        stds = [final_results[model][metric]['std'] for model in final_results]

        axes[i].barh(range(len(final_results)), means, xerr=stds)
        axes[i].set_yticks(range(len(final_results)))
        axes[i].set_yticklabels(final_results.keys(), fontsize=8)
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_prediction_scatter(all_predictions):
    n_models = len(all_predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, (model_name, predictions) in enumerate(all_predictions.items()):
        actuals, preds = zip(*predictions)
        axes[i].scatter(actuals, preds, alpha=0.5)
        axes[i].plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        axes[i].set_title(f'{model_name}\nR² = {final_results[model_name]["r2"]["mean"]:.3f}')
        axes[i].set_xlabel('Actual Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].grid(True)

    for i in range(len(all_predictions), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_error_distribution(all_predictions):
    n_models = len(all_predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, (model_name, predictions) in enumerate(all_predictions.items()):
        actuals, preds = zip(*predictions)
        errors = np.array(preds) - np.array(actuals)
        sns.histplot(errors, kde=True, ax=axes[i])
        axes[i].set_title(f'{model_name}\nError Distribution')
        axes[i].set_xlabel('Prediction Error')

    for i in range(len(all_predictions), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# Print detailed results
print("\nDetailed Cross-Validation Results:")
print("-" * 80)
for model_name, metrics in final_results.items():
    print(f"\n{model_name}:")
    for metric, values in metrics.items():
        print(f"{metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")

# Generate plots
plot_metrics_comparison(final_results)
plot_prediction_scatter(all_predictions)
plot_error_distribution(all_predictions)