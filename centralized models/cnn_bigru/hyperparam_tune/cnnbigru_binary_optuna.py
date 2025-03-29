#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter tuning for CNN+BiGRU model using Optuna
Focus: Make the model lightweight while maintaining good performance
"""

import os
import time
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load and preprocess data
def load_data(train_path, test_path):
    """Load and preprocess the UNSW-NB15 dataset for binary classification"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    data = pd.concat([train_df, test_df], ignore_index=True)

    # Define target variable
    binary_target = 'label'

    # Label encode categorical features
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Apply normalization to numerical features
    scaler = MinMaxScaler()
    features_to_scale = data.columns.difference([binary_target])
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Extract features and targets
    X = data.drop(columns=[binary_target])
    y_binary = data[binary_target]

    selected_features_indices = [0, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25, 28, 30, 31, 32, 33, 34, 35, 37, 40, 41]
    X_selected = X.iloc[:, selected_features_indices]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Split the data
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42)

    # Reshape data for CNN (adding channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train_binary, y_test_binary


# CNN+BiGRU model for binary classification with tunable hyperparameters
class CNNBiGRUBinaryModel(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, kernel_size=3, 
                 hidden_dim=50, fc_units=100, dropout=0.01,
                 bidirectional=True, num_layers=1):
        super(CNNBiGRUBinaryModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.fc_units = fc_units
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)

        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=cnn_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate input size for fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(fc_input_dim, fc_units)
        self.dropout2 = nn.Dropout(dropout)
        self.binary_output = nn.Linear(fc_units, 1)

    def forward(self, x):
        # Input shape: batch_size x seq_length x 1
        x = x.permute(0, 2, 1)  # Change to batch_size x 1 x seq_length for Conv1d

        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)

        # Change shape for GRU: batch_size x seq_length x channels
        x = x.permute(0, 2, 1)

        # Bidirectional GRU
        x, _ = self.bigru(x)

        # Global average pooling
        x = x.permute(0, 2, 1)  # Change to batch_size x channels x seq_length
        x = self.global_avg_pool(x).squeeze(-1)

        # Dense layers
        x = F.relu(self.fc(x))
        x = self.dropout2(x)

        # Output layer
        binary_output = torch.sigmoid(self.binary_output(x))

        return binary_output


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=10, patience=5):
    """Train the PyTorch model for binary classification"""
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    best_val_metrics = None

    # Training history
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            outputs = outputs.squeeze()

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = (outputs > 0.5).float()
            correct += (pred == targets).sum().item()
            total_samples += targets.size(0)

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

        # Calculate average training metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = correct / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0
        
        val_preds = []
        val_targets_list = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.float().to(device)

                # Forward
                outputs = model(inputs)
                outputs = outputs.squeeze()

                # Calculate loss
                loss = criterion(outputs, targets)

                # Calculate accuracy
                pred = (outputs > 0.5).float()
                val_correct += (pred == targets).sum().item()
                val_total_samples += targets.size(0)

                # Collect predictions and targets for metric calculation
                val_preds.extend(pred.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

                # Accumulate loss
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation metrics
        epoch_val_loss = val_loss / val_total_samples
        epoch_val_acc = val_correct / val_total_samples
        
        # Calculate additional validation metrics
        val_preds = np.array(val_preds)
        val_targets_list = np.array(val_targets_list)
        
        val_precision = precision_score(val_targets_list, val_preds)
        val_recall = recall_score(val_targets_list, val_preds)
        val_f1 = f1_score(val_targets_list, val_preds)

        # Update learning rate scheduler
        if scheduler:
            scheduler.step(epoch_val_loss)

        # Store metrics in history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        # Model checkpointing - save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
            
            # Save best validation metrics
            best_val_metrics = {
                'accuracy': epoch_val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            }
        else:
            early_stopping_counter += 1

        # Early stopping check
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            # Load the best model before returning
            model.load_state_dict(best_model_state)
            break

    # Ensure we use the best model
    if best_model_state is not None and early_stopping_counter < patience:
        model.load_state_dict(best_model_state)

    return history, best_val_metrics


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Load data only once per trial
    if not hasattr(objective, "data_loaded"):
        # Set paths to your data files
        train_path = '../../../dataset/UNSW_NB15_testing-set.csv'
        test_path = '../../../dataset/UNSW_NB15_testing-set.csv'
        
        # Load and preprocess data
        X_train, X_test, y_train_binary, y_test_binary = load_data(train_path, test_path)
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_binary_tensor = torch.FloatTensor(y_train_binary.values)
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_binary_tensor = torch.FloatTensor(y_test_binary.values)
        
        # Store data for reuse
        objective.X_train = X_train
        objective.X_test = X_test
        objective.X_train_tensor = X_train_tensor
        objective.y_train_binary_tensor = y_train_binary_tensor
        objective.X_test_tensor = X_test_tensor
        objective.y_test_binary_tensor = y_test_binary_tensor
        objective.data_loaded = True
    
    # Access data
    X_train = objective.X_train
    X_train_tensor = objective.X_train_tensor
    y_train_binary_tensor = objective.y_train_binary_tensor
    
    # Define hyperparameters to tune
    cnn_filters = trial.suggest_categorical("cnn_filters", [8, 16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 50])
    fc_units = trial.suggest_categorical("fc_units", [16, 32, 64, 100])
    dropout = trial.suggest_float("dropout", 0.01, 0.3)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_binary_tensor)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = CNNBiGRUBinaryModel(
        input_dim=input_dim,
        cnn_filters=cnn_filters,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        fc_units=fc_units,
        dropout=dropout,
        bidirectional=bidirectional,
        num_layers=num_layers
    ).to(device)
    
    # Count model parameters
    model_size = count_parameters(model)
    print(f"Trial {trial.number}: Model Parameters: {model_size}")
    
    # Set up loss function, optimizer and scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Train the model
    history, best_val_metrics = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=15,  # Reduced for tuning
        patience=3       # Reduced for tuning
    )
    
    # Calculate the combined score: performance + lightweight bonus
    # We prioritize accuracy and F1 score while penalizing large models
    accuracy = best_val_metrics['accuracy']
    f1 = best_val_metrics['f1']
    
    # Model size penalty (normalized by original model size ~100k params)
    original_model_size = 100000
    size_ratio = model_size / original_model_size
    
    # Combined score: 0.5*accuracy + 0.5*f1 - (size penalty)
    # The size_penalty_factor can be adjusted to weight model size importance
    size_penalty_factor = 0.05
    size_penalty = size_penalty_factor * size_ratio
    
    # We limit the penalty to a reasonable range
    size_penalty = min(size_penalty, 0.1)
    
    # Calculate combined score with more focus on accuracy and F1 while considering model size
    combined_score = 0.5 * accuracy + 0.5 * f1 - size_penalty
    
    # Log parameters and results
    print(f"Trial {trial.number}:")
    print(f"  CNN Filters: {cnn_filters}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  FC Units: {fc_units}")
    print(f"  Bidirectional: {bidirectional}")
    print(f"  Num Layers: {num_layers}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Model Size: {model_size} parameters")
    print(f"  Size Penalty: {size_penalty:.4f}")
    print(f"  Combined Score: {combined_score:.4f}")
    
    # Set additional attributes for analysis
    trial.set_user_attr("accuracy", float(accuracy))
    trial.set_user_attr("f1", float(f1))
    trial.set_user_attr("model_size", int(model_size))
    trial.set_user_attr("combined_score", float(combined_score))
    
    return combined_score


def save_best_model(study, X_train, X_test, y_train_binary, y_test_binary, output_dir="binary_model_hyperopt"):
    """Train and save the best model from hyperparameter optimization"""
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get best trial hyperparameters
    best_trial = study.best_trial
    params = best_trial.params
    
    # Create PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_binary_tensor = torch.FloatTensor(y_train_binary.values)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_binary_tensor = torch.FloatTensor(y_test_binary.values)
    
    # Create datasets and dataloaders
    batch_size = params["batch_size"]
    train_dataset = TensorDataset(X_train_tensor, y_train_binary_tensor)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_binary_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model with best hyperparameters
    input_dim = X_train.shape[1]
    best_model = CNNBiGRUBinaryModel(
        input_dim=input_dim,
        cnn_filters=params["cnn_filters"],
        kernel_size=params["kernel_size"],
        hidden_dim=params["hidden_dim"],
        fc_units=params["fc_units"], 
        dropout=params["dropout"],
        bidirectional=params["bidirectional"],
        num_layers=params["num_layers"]
    ).to(device)
    
    # Print model summary
    print("\nBest Model Architecture:")
    print(best_model)
    print(f"\nTotal Parameters: {count_parameters(best_model)}")
    
    # Set up loss function, optimizer and scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Train the best model with more epochs
    history, _ = train_model(
        best_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=30,
        patience=5
    )
    
    # Save best model
    best_model_path = os.path.join(output_dir, "best_model.pth")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'hyperparameters': params,
        'input_dim': input_dim,
    }, best_model_path)
    print(f"Saved best model to {best_model_path}")
    
    # Save hyperparameters
    import json
    with open(os.path.join(output_dir, "best_hyperparameters.json"), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Evaluate on test data
    metrics, conf_matrix = evaluate_model(best_model, test_loader)
    
    # Save evaluation metrics
    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    # Plot test metrics
    plot_metrics(metrics, output_dir)
    
    return best_model, metrics


def evaluate_model(model, test_loader):
    """Evaluate the binary classification model on test data"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            # Forward
            outputs = model(inputs)
            
            # Convert outputs to predictions
            preds = (outputs > 0.5).int().cpu().numpy()
            
            # Store predictions and targets
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Print classification metrics
    print(f'\nTest Set Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    # Store the metrics
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    return metrics, conf_matrix


def plot_metrics(metrics, output_dir="binary_model_hyperopt"):
    """Plot the evaluation metrics for binary classification"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics[cat] for cat in categories]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color='#003566')
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Lightweight CNN+BiGRU Model - Binary Classification')
    plt.ylim(0, 1.0)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_metrics.png'))
    plt.close()


def plot_optimization_history(study, output_dir="binary_model_hyperopt"):
    """Plot the optimization history from Optuna study"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    
    # Extract data
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    accuracies = [t.user_attrs["accuracy"] for t in trials if "accuracy" in t.user_attrs]
    f1_scores = [t.user_attrs["f1"] for t in trials if "f1" in t.user_attrs]
    model_sizes = [t.user_attrs["model_size"] for t in trials if "model_size" in t.user_attrs]
    
    # Plot combined score
    plt.subplot(2, 2, 1)
    plt.plot(values, 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel('Combined Score')
    plt.title('Optimization History')
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(accuracies, 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(f1_scores, 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel('F1 Score')
    plt.title('F1 Score History')
    
    # Plot model size
    plt.subplot(2, 2, 4)
    plt.plot(model_sizes, 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel('Model Parameters')
    plt.title('Model Size History')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
    plt.close()
    
    # Plot parameter importances
    try:
        param_importances = optuna.importance.get_param_importances(study)
        importance_values = list(param_importances.values())
        importance_names = list(param_importances.keys())
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_names, importance_values)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_importance.png'))
        plt.close()
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")
    
    # Plot parallel coordinate plot for visualization
    try:
        from optuna.visualization import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
    except Exception as e:
        print(f"Could not create parallel coordinate plot: {e}")


def main():
    # Output directory
    output_dir = "../binary_model_hyperopt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set paths to your data files
    train_path = '../../../dataset/UNSW_NB15_testing-set.csv'
    test_path = '../../../dataset/UNSW_NB15_testing-set.csv'
    
    print("Starting hyperparameter optimization for lightweight CNN+BiGRU model...")
    
    # Create an Optuna study that maximizes the score
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="lightweight_cnn_bigru"
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=25, timeout=7200)  # 25 trials or 2 hours
    
    # Print optimization results
    print("\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Load data for saving the best model
    X_train, X_test, y_train_binary, y_test_binary = load_data(train_path, test_path)
    
    # Save the best model
    best_model, metrics = save_best_model(
        study, X_train, X_test, y_train_binary, y_test_binary, output_dir
    )
    
    # Plot optimization history
    plot_optimization_history(study, output_dir)
    
    # Compare to original model
    print("\nModel Size Comparison:")
    original_model = CNNBiGRUBinaryModel(input_dim=X_train.shape[1], hidden_dim=50)
    original_size = count_parameters(original_model)
    best_size = count_parameters(best_model)
    size_reduction = (original_size - best_size) / original_size * 100
    
    print(f"  Original Model Size: {original_size} parameters")
    print(f"  Optimized Model Size: {best_size} parameters")
    print(f"  Size Reduction: {size_reduction:.2f}%")
    print("\nPerformance:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    
    # Save comparison results
    comparison = {
        "original_model_size": original_size,
        "optimized_model_size": best_size,
        "size_reduction_percent": float(size_reduction),
        "accuracy": float(metrics['Accuracy']),
        "f1_score": float(metrics['F1 Score']),
    }
    
    import json
    with open(os.path.join(output_dir, "model_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=4)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal optimization time: {(end_time - start_time) / 60:.2f} minutes") 