#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN+BiGRU model for binary intrusion detection on UNSW-NB15 dataset
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as f
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


# CNN+BiGRU model for binary classification
class CNNBiGRUBinaryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.01):
        super(CNNBiGRUBinaryModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)

        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # Output layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, 100)
        self.dropout2 = nn.Dropout(dropout)
        self.binary_output = nn.Linear(100, 1)

    def forward(self, x):
        # Input shape: batch_size x seq_length x 1
        x = x.permute(0, 2, 1)  # Change to batch_size x 1 x seq_length for Conv1d

        # CNN layers
        x = f.relu(self.conv1(x))
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
        x = f.relu(self.fc(x))
        x = self.dropout2(x)

        # Output layer
        binary_output = torch.sigmoid(self.binary_output(x))

        return binary_output


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, patience=5,
                checkpoint_dir="binary_model_checkpoints"):
    """Train the PyTorch model for binary classification"""
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )

    # Early stopping variables
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    # Training history
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'learning_rates': []}

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

                # Accumulate loss
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation metrics
        epoch_val_loss = val_loss / val_total_samples
        epoch_val_acc = val_correct / val_total_samples

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

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

            # Save best model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'accuracy': epoch_val_acc
            }, checkpoint_path)
            print(f"Saved new best model checkpoint with validation loss: {best_val_loss: .4f}")
        else:
            early_stopping_counter += 1

        # Save epoch checkpoint (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': epoch_val_loss,
            }, epoch_checkpoint_path)

        # Early stopping check
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            # Load the best model before returning
            model.load_state_dict(best_model_state)
            break

        # Print progress
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {epoch_loss: .4f}, '
              f'Acc: {epoch_acc: .4f}, '
              f'Val Loss: {epoch_val_loss: .4f}, '
              f'Val Acc: {epoch_val_acc: .4f}, '
              f'LR: {current_lr: .6f}')

    # Ensure we return the best model
    if best_model_state is not None and early_stopping_counter < patience:
        model.load_state_dict(best_model_state)
        print("Loaded best model from checkpoints")

    return history


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
    print(f'Binary Classification Metrics:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix: \n{conf_matrix}')

    # Store the metrics
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    return metrics, conf_matrix


def plot_metrics(metrics):
    """Plot the evaluation metrics for binary classification"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics[cat] for cat in categories]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color='#003566')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('CNN + Bi-GRU with Attention - Binary Classification (PyTorch)')
    plt.ylim(0, 1.0)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('binary_model_metrics.png')
    plt.show()


def main():
    # Set paths to your data files
    train_path = '../../dataset/UNSW_NB15_testing-set.csv'
    test_path = '../../dataset/UNSW_NB15_testing-set.csv'

    # Create model checkpoint directory
    checkpoint_dir = "binary_model_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load and preprocess data
    X_train, X_test, y_train_binary, y_test_binary = load_data(train_path, test_path)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_binary_tensor = torch.FloatTensor(y_train_binary.values)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_binary_tensor = torch.FloatTensor(y_test_binary.values)

    # Create PyTorch datasets and dataloaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_binary_tensor)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(X_test_tensor, y_test_binary_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    input_dim = X_train.shape[1]
    hidden_dim = 50
    model = CNNBiGRUBinaryModel(input_dim, hidden_dim, dropout=0.01).to(device)
    print(model)

    # Check if we want to resume training from a checkpoint
    resume_training = False  # Set to True if you want to resume from a checkpoint
    resume_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if resume_training and os.path.exists(resume_checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resuming training from epoch {checkpoint['epoch']}")

    # Train model with early stopping, checkpointing, and learning rate scheduling
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=30,
        learning_rate=0.001,
        patience=5,  # Early stopping patience
        checkpoint_dir=checkpoint_dir
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Plot training history
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Binary Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'binary_training_history.png'))
    plt.show()

    # Evaluate model on test set
    # Load the best model for evaluation
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} for evaluation")

    # Evaluate model
    metrics, confusion_matrix = evaluate_model(model, test_loader)

    # Plot metrics
    plot_metrics(metrics)

    # Save evaluation results
    import json
    with open(os.path.join(checkpoint_dir, 'binary_evaluation_metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)


if __name__ == "__main__":
    main() 