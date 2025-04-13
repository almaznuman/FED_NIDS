"""BiGRU: A Flower / PyTorch app for anomaly detection."""

import pandas as pd
from datasets import load_dataset
from flwr.common.typing import UserConfig
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
# from flwr_datasets.visualization import plot_label_distributions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import numpy as np

# Define feature dimensions
features = [0, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25, 28, 30, 31, 32, 33, 34, 35, 37, 40, 41]


# CNN+BiGRU model for binary classification
class CNNBiGRUBinaryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, dropout=0.01):
        super(CNNBiGRUBinaryModel, self).__init__()
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


def load_model(learning_rate: float = 0.001):
    """Create and initialize a new CNN+BiGRU model.
    
    Args:
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary containing model, optimizer, and criterion
    """
    # Create model but defer device placement to when it's actually used
    model = CNNBiGRUBinaryModel(input_dim=len(features))
    # Remove weight_decay to match centralized implementation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion
    }


def get_weights(net):
    """Get model weights as a list of NumPy arrays.
    
    Args:
        net: PyTorch model
        
    Returns:
        List of NumPy arrays representing model weights
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays.
    
    Args:
        net: PyTorch model
        parameters: List of NumPy arrays representing model weights
    """
    # Get the device of the model once to avoid repeated checks
    device = next(net.parameters()).device
    
    # Process all parameters at once with device handling
    state_dict = OrderedDict(
        {k: torch.tensor(v, device=device) for k, v in zip(net.state_dict().keys(), parameters)}
    )
    
    # Load state dict
    net.load_state_dict(state_dict, strict=True)


# Cache for FederatedDataset
fds = None


def preprocess_data(data):
    """Common preprocessing function for both training and test data.
    
    Args:
        data: DataFrame containing the data to preprocess
        
    Returns:
        tuple: (X_scaled, y) preprocessed features and labels
    """
    # Drop unnecessary columns if they exist
    if "id" in data.columns and "attack_cat" in data.columns:
        data.drop(columns=["id", "attack_cat"], inplace=True)
    elif "id" in data.columns:
        data.drop(columns=["id"], inplace=True)

    # Label encode categorical features
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Apply MinMax normalization to all features
    scaler = MinMaxScaler()
    features_to_scale = data.columns.difference(['label'])
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Extract features and targets
    X = data.drop(columns=['label'])
    y = data['label']

    # Select features based on the predefined indices
    X_selected = X.iloc[:, features]

    # Standardize the selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Reshape for CNN input
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    return X_scaled, y


def load_data(partition_id, num_partitions, alpha):
    """Load and preprocess data for the specified partition.
    
    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        alpha: Heterogeneity factor
        
    Returns:
        tuple: (train_loader, test_loader) with DataLoader objects
    """
    global fds

    # Load dataset only once
    if fds is None:
        data_files = "dataset/UNSW_NB15_training-set.csv"
        dataset = load_dataset("csv", data_files=data_files)

        # Initialize the partitioner
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",  # Your target column
            alpha=alpha,
            self_balancing=False,
            seed=42
        )

        # Use IidPartitioner instead
        # partitioner = IidPartitioner(
        #     num_partitions=num_partitions
        # )

        partitioner.dataset = dataset["train"]
        fds = partitioner  # Cache the partitioner

    # Load partition
    partition = fds.load_partition(partition_id=partition_id)
    data = pd.DataFrame(partition)
    
    # Use common preprocessing function
    X_scaled, y = preprocess_data(data)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader


def load_cen_data():
    """Load and preprocess the entire centralized test dataset for model evaluation.
    
    Returns:
        DataLoader for the centralized test dataset
    """
    # Load the full test dataset
    data_files = "dataset/UNSW_NB15_testing-set.csv"
    dataset = load_dataset("csv", data_files=data_files)

    # Convert to Pandas DataFrame
    data = pd.DataFrame(dataset['train'])
    
    # Use common preprocessing function
    X_scaled, y = preprocess_data(data)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return dataloader


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    # save_path = Path.cwd() / f"outputs_poc/{run_dir}"
    # save_path = Path.cwd() / f"outputs_fedavg/{run_dir}"
    # save_path = Path.cwd() / f"outputs_poc_low/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


def train(net, trainloader, epochs, optimizer, criterion, device):
    """Train the model.
    
    Args:
        net: PyTorch model
        trainloader: DataLoader for training data
        epochs: Number of epochs to train
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on (cuda/cpu)
        
    Returns:
        float: Average training loss
    """
    # Move model and criterion to device once
    net = net.to(device)
    criterion = criterion.to(device)
    net.train()
    
    # Add LR scheduler to match centralized implementation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Early stopping variables
    best_loss = float('inf')
    early_stopping_counter = 0
    patience = 5  # Same as centralized
    best_model_state = None
    
    running_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, targets.squeeze())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            batch_count += 1
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / batch_count
        
        # Update the scheduler based on validation loss
        scheduler.step(avg_epoch_loss)
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = net.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            # Load the best model
            if best_model_state is not None:
                net.load_state_dict(best_model_state)
            break
        
        running_loss += avg_epoch_loss
        total_batches += 1
    
    return running_loss / total_batches


def test(net, testloader, criterion, device):
    """Test the model.
    
    Args:
        net: PyTorch model
        testloader: DataLoader for test data
        criterion: Loss function
        device: Device to test on (cuda/cpu)
        
    Returns:
        tuple: (average loss, predictions, actual labels)
    """
    # Move model and criterion to device once
    net = net.to(device)
    criterion = criterion.to(device)
    net.eval()
    running_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = net(inputs)
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, targets.squeeze())
            
            # Update statistics
            running_loss += loss.item()
            total_batches += 1
            
            # Store predictions and targets
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    return running_loss / total_batches, torch.tensor(all_predictions), torch.tensor(all_targets)
