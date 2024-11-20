import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from itertools import product
from collections import defaultdict
import time
import scienceplots

plt.style.use(['science', 'ieee'])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

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
        if len(self.loss_history) < 3:
            return 0

        diffs = np.diff(self.loss_history[-self.plateau_window:])
        oscillations = 0
        for i in range(len(diffs) - 1):
            if (abs(diffs[i]) > self.oscillation_threshold and
                    abs(diffs[i + 1]) > self.oscillation_threshold and
                    diffs[i] * diffs[i + 1] < 0):
                oscillations += 1

        return oscillations

    def check_plateau(self):
        if len(self.loss_history) < self.plateau_window:
            return False

        recent_losses = np.array(self.loss_history[-self.plateau_window:])
        num_oscillations = self.count_oscillations()
        start_loss = np.mean(recent_losses[:3])
        end_loss = np.mean(recent_losses[-3:])
        relative_improvement = abs((start_loss - end_loss) / start_loss) if start_loss != 0 else 0

        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        coefficient_of_variation = loss_std / loss_mean if loss_mean != 0 else 0

        return ((num_oscillations >= self.max_oscillations and relative_improvement < self.oscillation_threshold) or
                (coefficient_of_variation < self.oscillation_threshold and len(recent_losses) >= self.plateau_window))

    def __call__(self, val_loss, train_loss, model):
        self.loss_history.append(val_loss)

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_model(model)
            return False

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


def train_evaluate_traditional_models(X, y, n_splits=5):
    """Train and evaluate traditional ML models using cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'LinearRegression': LinearRegression()
    }

    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_val_scaled)

            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'mae': mean_absolute_error(y_val, y_pred),
                'mape': mean_absolute_percentage_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }

            # Store results
            for metric_name, value in metrics.items():
                results[name][metric_name].append(value)

    # Calculate final results
    final_results = {}
    for model_name in results:
        final_results[model_name] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in results[model_name].items()
        }

    return final_results


def format_results_table(final_results):
    """Create a formatted DataFrame of results"""
    rows = []
    for model_name, metrics in final_results.items():
        row = {
            'Model': model_name,
            'MSE': f"{metrics['mse']['mean']:.4f} ± {metrics['mse']['std']:.4f}",
            'RMSE': f"{metrics['rmse']['mean']:.4f} ± {metrics['rmse']['std']:.4f}",
            'MAE': f"{metrics['mae']['mean']:.4f} ± {metrics['mae']['std']:.4f}",
            'MAPE': f"{metrics['mape']['mean']:.4f} ± {metrics['mape']['std']:.4f}",
            'R²': f"{metrics['r2']['mean']:.4f} ± {metrics['r2']['std']:.4f}"
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index('Model')


def find_best_architectures(final_results):
    """Find the best architecture for each activation function based on MSE"""
    best_archs = {}
    for model_name, metrics in final_results.items():
        if not model_name.startswith('MLP'):
            continue
        act_fn = model_name.split('_')[1]
        mse = metrics['mse']['mean']

        if act_fn not in best_archs or mse < best_archs[act_fn]['mse']:
            best_archs[act_fn] = {
                'architecture': '_'.join(model_name.split('_')[2:]),
                'mse': mse,
                'full_name': model_name,
                'metrics': metrics
            }

    return best_archs


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
    plt.figure(figsize=(15, 8))
    mse_means = [results['mse']['mean'] for results in final_results.values()]
    mse_stds = [results['mse']['std'] for results in final_results.values()]
    model_names = list(final_results.keys())

    plt.barh(range(len(model_names)), mse_means, xerr=mse_stds, height=0.8)
    plt.yticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], fontsize=8)
    plt.xlabel('Mean Squared Error')
    plt.title('Model Comparison - MSE (lower is better)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_detailed_results(final_results, training_histories, output_file='detailed_model_results.csv'):
    """
    Save detailed results including model name, metrics, and training characteristics
    """
    detailed_results = []

    for model_name, metrics in final_results.items():
        # Get training histories for this model
        histories = training_histories.get(model_name, [])

        # Aggregate training history information
        avg_epochs = np.mean([h['epochs_run'] for h in histories]) if histories else None
        stop_reasons = [h['stop_reason'] for h in histories] if histories else []
        stop_reason_counts = pd.Series(stop_reasons).value_counts().to_dict() if stop_reasons else {}

        # Prepare a row with all information
        row = {
            'Model Name': model_name,
            'MSE Mean': metrics['mse']['mean'],
            'MSE Std': metrics['mse']['std'],
            'RMSE Mean': metrics['rmse']['mean'],
            'RMSE Std': metrics['rmse']['std'],
            'MAE Mean': metrics['mae']['mean'],
            'MAE Std': metrics['mae']['std'],
            'MAPE Mean': metrics['mape']['mean'],
            'MAPE Std': metrics['mape']['std'],
            'R2 Mean': metrics['r2']['mean'],
            'R2 Std': metrics['r2']['std'],
            'Avg Epochs': avg_epochs,
            **{f'Stop Reason {k}': v for k, v in stop_reason_counts.items()}
        }

        detailed_results.append(row)

    # Convert to DataFrame and save
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(output_file, index=False)
    return detailed_df


def plot_best_architectures_performance(best_architectures, final_results):
    """
    Create a bar plot comparing the best architecture for each activation function
    """
    plt.figure(figsize=(12, 6))

    # Prepare data for plotting
    act_fns = []
    mse_means = []
    mse_stds = []

    for act_fn, info in best_architectures.items():
        act_fns.append(act_fn)
        mse_means.append(info['metrics']['mse']['mean'])
        mse_stds.append(info['metrics']['mse']['std'])

    # Create horizontal bar plot with error bars
    plt.barh(act_fns, mse_means, xerr=mse_stds, capsize=5)
    plt.xlabel('Mean Squared Error')
    plt.title('Best Architecture Performance by Activation Function')
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


def print_detailed_best_architectures(best_architectures, final_results):
    """
    Print comprehensive details for the best architecture of each activation function
    """
    print("\nDetailed Best Architectures for Each Activation Function:")
    print("-" * 50)
    for act_fn, info in best_architectures.items():
        print(f"\n{act_fn} Activation Function:")
        print(f"  Best Architecture: {info['architecture']}")
        print("  Comprehensive Metrics:")

        # Get full metrics for this model
        metrics = final_results[info['full_name']]

        # Print each metric with mean and standard deviation
        metric_names = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in metric_names:
            print(f"    {metric.upper()}: {metrics[metric]['mean']:.4f} ± {metrics[metric]['std']:.4f}")


def plot_predictions_scatter(all_predictions, best_model):
    """
    Create a scatter plot of predicted vs actual values for the best model

    Parameters:
    - all_predictions: Dictionary containing predictions for all models
    - best_model: Name of the best performing model
    """
    # Extract actual and predicted values for the best model
    model_predictions = all_predictions[best_model]
    actuals, predictions = zip(*model_predictions)

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals, predictions, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)

    # Add perfect prediction line
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--',
             label='Perfect Prediction')

    # Calculate R-squared
    r2 = r2_score(actuals, predictions)

    # Compute additional metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)

    # Annotate the plot with performance metrics
    plt.title(f'Predicted vs Actual Values\n{best_model}')
    plt.xlabel('Actual Concrete Compressive Strength')
    plt.ylabel('Predicted Concrete Compressive Strength')

    # Add metrics to the plot
    plt.annotate(f'R² = {r2:.4f}\n'
                 f'MSE = {mse:.4f}\n'
                 f'RMSE = {rmse:.4f}\n'
                 f'MAE = {mae:.4f}\n'
                 f'MAPE = {mape:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 verticalalignment='top', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define architectures and activation functions
    architectures = [
        [64, 32],
        [128, 64, 32],
        [256, 128, 64, 32],
        # [512, 256, 128, 64, 32],
        # [32, 32],
        [64]
    ]

    activation_fns = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'GELU': nn.GELU,
        # 'Tanh': nn.Tanh
    }

    # Cross-validation setup
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary to store results
    cv_results = defaultdict(lambda: defaultdict(list))
    training_histories = defaultdict(list)
    all_predictions = defaultdict(list)

    # Perform cross-validation for MLPs
    print("Training MLP models...")
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
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

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

        print(f"Fold {fold + 1} time taken: {time.time() - startTime:.2f} seconds")

    # Calculate final results for MLPs
    mlp_final_results = {}
    for model_name in cv_results:
        mlp_final_results[model_name] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in cv_results[model_name].items()
        }

    # Train and evaluate traditional models
    print("\nTraining traditional models...")
    traditional_results = train_evaluate_traditional_models(X, y)

    # Combine all results
    all_results = {**mlp_final_results, **traditional_results}

    # Find the best architectures for each activation function
    best_architectures = find_best_architectures(mlp_final_results)

    # Print results
    print("\nBest Architecture for Each Activation Function:")
    print("-" * 50)
    for act_fn, info in best_architectures.items():
        print(f"{act_fn}:")
        print(f"  Architecture: {info['architecture']}")
        print(f"metrics: {info['metrics']}")
        # print(f"  MSE: {info['mse']:.4f}")
        print()

    # Print detailed results
    print("\nDetailed Cross-Validation Results:")
    print("-" * 80)
    results_table = format_results_table(all_results)
    print(results_table)
    results_table.to_csv('cross_validation_results.csv')

    # Plot comparisons
    print("\nGenerating plots...")

    # Plot overall model comparison
    plot_metrics_comparison(all_results)

    # Plot training curves for best model
    best_model = min(mlp_final_results.items(), key=lambda x: x[1]['mse']['mean'])[0]
    print(f"\nPlotting training curves for best model: {best_model}")

    # Plot training curves for each fold of the best model
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

    # Additional analysis: Learning stability
    print("\nLearning Stability Analysis:")
    print("-" * 50)
    for model_name in mlp_final_results.keys():
        avg_epochs = np.mean([h['epochs_run'] for h in training_histories[model_name]])
        stop_reasons = [h['stop_reason'] for h in training_histories[model_name]]
        stop_reason_counts = pd.Series(stop_reasons).value_counts()



        print(f"\n{model_name}:")
        print(f"  Average epochs until convergence: {avg_epochs:.1f}")
        print("  Stop reasons distribution:")
        for reason, count in stop_reason_counts.items():
            print(f"    - {reason}: {count}/{n_splits} folds")

    # Save best model configuration
    best_config = {
        'model_name': best_model,
        'architecture': best_architectures[best_model.split('_')[1]]['architecture'],
        'activation': best_model.split('_')[1],
        'mse': mlp_final_results[best_model]['mse']['mean'],
        'r2': mlp_final_results[best_model]['r2']['mean']
    }

    print("\nBest Overall Model Configuration:")
    print("-" * 50)
    print(f"Model: {best_config['model_name']}")
    print(f"Architecture: {best_config['architecture']}")
    print(f"Activation Function: {best_config['activation']}")
    print(f"MSE: {best_config['mse']:.4f}")
    print(f"R²: {best_config['r2']:.4f}")

    # Save detailed results to CSV
    detailed_results_df = save_detailed_results(
        all_results,
        training_histories,
        output_file='comprehensive_model_results23.csv'
    )

    # Print detailed metrics for best architectures
    print_detailed_best_architectures(best_architectures, all_results)

    # Plot performance of best architectures
    plot_best_architectures_performance(best_architectures, all_results)

    print("\nDetailed results and plots have been generated and saved.")

    # Plot scatter plot for the best model
    print(f"\nGenerating scatter plot for best model: {best_model}")
    plot_predictions_scatter(all_predictions, best_model)

    print("\nScatter plot of predicted vs actual values has been generated.")