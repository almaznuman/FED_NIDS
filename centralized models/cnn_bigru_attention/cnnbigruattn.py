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
    """Load and preprocess the UNSW-NB15 dataset"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    data = pd.concat([train_df, test_df], ignore_index=True)

    # Define target variables
    categorical_target = 'attack_cat'
    binary_target = 'label'

    # Label encode categorical features
    label_encoder = LabelEncoder()
    if categorical_target in data.columns:
        data[categorical_target] = label_encoder.fit_transform(data[categorical_target])

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Apply normalization to numerical features
    scaler = MinMaxScaler()
    features_to_scale = data.columns.difference([categorical_target, binary_target])
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Extract features and targets
    X = data.drop(columns=[binary_target, categorical_target])
    y_binary = data[binary_target]
    y_multi_class = data[categorical_target]

    selected_features_indices = [0, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25, 28, 30, 31, 32, 33, 34, 35, 37, 40, 41]
    X_selected = X.iloc[:, selected_features_indices]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Split the data
    X_train, X_test, y_train_binary, y_test_binary, y_train_multi_class, y_test_multi_class = train_test_split(
        X_scaled, y_binary, y_multi_class, test_size=0.2, random_state=42)

    # Reshape data for CNN (adding channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train_binary, y_test_binary, y_train_multi_class, y_test_multi_class


# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        out = self.layer_norm(x + attn_output)
        return out


# CNN+BiGRU model with attention
class CNNBiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.01):
        super(CNNBiGRUModel, self).__init__()
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

        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads=2)

        # Output layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, 100)
        self.dropout2 = nn.Dropout(dropout)
        self.binary_output = nn.Linear(100, 1)
        self.multiclass_output = nn.Linear(100, num_classes)

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

        # Attention mechanism
        x = self.attention(x)

        # Global average pooling
        x = x.permute(0, 2, 1)  # Change to batch_size x channels x seq_length
        x = self.global_avg_pool(x).squeeze(-1)

        # Dense layers
        x = f.relu(self.fc(x))
        x = self.dropout2(x)

        # Output layers
        binary_output = torch.sigmoid(self.binary_output(x))
        multiclass_output = f.softmax(self.multiclass_output(x), dim=1)

        return binary_output, multiclass_output


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, patience=5,
                checkpoint_dir="model_checkpoints"):
    """Train the PyTorch model"""
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Loss functions
    binary_criterion = nn.BCELoss()
    multiclass_criterion = nn.CrossEntropyLoss()

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
    history = {'binary_loss': [], 'multiclass_loss': [], 'total_loss': [],
               'binary_acc': [], 'multiclass_acc': [],
               'val_binary_loss': [], 'val_multiclass_loss': [], 'val_total_loss': [],
               'val_binary_acc': [], 'val_multiclass_acc': [],
               'learning_rates': []}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_binary_loss = 0.0
        running_multiclass_loss = 0.0
        running_total_loss = 0.0
        binary_correct = 0
        multiclass_correct = 0
        total_samples = 0

        for inputs, binary_targets, multiclass_targets in train_loader:
            inputs = inputs.to(device)
            binary_targets = binary_targets.float().to(device)
            multiclass_targets = multiclass_targets.long().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            binary_outputs, multiclass_outputs = model(inputs)
            binary_outputs = binary_outputs.squeeze()

            # Calculate losses
            binary_loss = binary_criterion(binary_outputs, binary_targets)
            multiclass_loss = multiclass_criterion(multiclass_outputs, multiclass_targets)
            total_loss = binary_loss + multiclass_loss

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Calculate accuracy
            binary_pred = (binary_outputs > 0.5).float()
            binary_correct += (binary_pred == binary_targets).sum().item()

            multiclass_pred = torch.argmax(multiclass_outputs, dim=1)
            multiclass_correct += (multiclass_pred == multiclass_targets).sum().item()

            total_samples += binary_targets.size(0)

            # Accumulate loss
            running_binary_loss += binary_loss.item() * inputs.size(0)
            running_multiclass_loss += multiclass_loss.item() * inputs.size(0)
            running_total_loss += total_loss.item() * inputs.size(0)

        # Calculate average training metrics
        epoch_binary_loss = running_binary_loss / total_samples
        epoch_multiclass_loss = running_multiclass_loss / total_samples
        epoch_total_loss = running_total_loss / total_samples
        epoch_binary_acc = binary_correct / total_samples
        epoch_multiclass_acc = multiclass_correct / total_samples

        # Validation
        model.eval()
        val_binary_loss = 0.0
        val_multiclass_loss = 0.0
        val_total_loss = 0.0
        val_binary_correct = 0
        val_multiclass_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, binary_targets, multiclass_targets in val_loader:
                inputs = inputs.to(device)
                binary_targets = binary_targets.float().to(device)
                multiclass_targets = multiclass_targets.long().to(device)

                # Forward
                binary_outputs, multiclass_outputs = model(inputs)
                binary_outputs = binary_outputs.squeeze()

                # Calculate losses
                binary_loss = binary_criterion(binary_outputs, binary_targets)
                multiclass_loss = multiclass_criterion(multiclass_outputs, multiclass_targets)
                total_loss = binary_loss + multiclass_loss

                # Calculate accuracy
                binary_pred = (binary_outputs > 0.5).float()
                val_binary_correct += (binary_pred == binary_targets).sum().item()

                multiclass_pred = torch.argmax(multiclass_outputs, dim=1)
                val_multiclass_correct += (multiclass_pred == multiclass_targets).sum().item()

                val_total_samples += binary_targets.size(0)

                # Accumulate loss
                val_binary_loss += binary_loss.item() * inputs.size(0)
                val_multiclass_loss += multiclass_loss.item() * inputs.size(0)
                val_total_loss += total_loss.item() * inputs.size(0)

        # Calculate average validation metrics
        epoch_val_binary_loss = val_binary_loss / val_total_samples
        epoch_val_multiclass_loss = val_multiclass_loss / val_total_samples
        epoch_val_total_loss = val_total_loss / val_total_samples
        epoch_val_binary_acc = val_binary_correct / val_total_samples
        epoch_val_multiclass_acc = val_multiclass_correct / val_total_samples

        # Update learning rate scheduler
        scheduler.step(epoch_val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        # Store metrics in history
        history['binary_loss'].append(epoch_binary_loss)
        history['multiclass_loss'].append(epoch_multiclass_loss)
        history['total_loss'].append(epoch_total_loss)
        history['binary_acc'].append(epoch_binary_acc)
        history['multiclass_acc'].append(epoch_multiclass_acc)

        history['val_binary_loss'].append(epoch_val_binary_loss)
        history['val_multiclass_loss'].append(epoch_val_multiclass_loss)
        history['val_total_loss'].append(epoch_val_total_loss)
        history['val_binary_acc'].append(epoch_val_binary_acc)
        history['val_multiclass_acc'].append(epoch_val_multiclass_acc)

        # Model checkpointing - save best model
        if epoch_val_total_loss < best_val_loss:
            best_val_loss = epoch_val_total_loss
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
                'binary_acc': epoch_val_binary_acc,
                'multiclass_acc': epoch_val_multiclass_acc
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
                'val_loss': epoch_val_total_loss,
            }, epoch_checkpoint_path)

        # Early stopping check
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            # Load the best model before returning
            model.load_state_dict(best_model_state)
            break

        # Print progress
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Loss: {epoch_total_loss: .4f}, '
              f' Binary Acc: {epoch_binary_acc: .4f}, '
              f' Multiclass Acc: {epoch_multiclass_acc: .4f}, '
              f'Val Loss: {epoch_val_total_loss: .4f}, '
              f' Val Binary Acc: {epoch_val_binary_acc: .4f}, '
              f' Val Multiclass Acc: {epoch_val_multiclass_acc: .4f}, '
              f'LR: {current_lr: .6f}')

    # Ensure we return the best model
    if best_model_state is not None and early_stopping_counter < patience:
        model.load_state_dict(best_model_state)
        print("Loaded best model from checkpoints")

    return history


def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()

    all_binary_preds = []
    all_multiclass_preds = []
    all_binary_targets = []
    all_multiclass_targets = []

    with torch.no_grad():
        for inputs, binary_targets, multiclass_targets in test_loader:
            inputs = inputs.to(device)

            # Forward
            binary_outputs, multiclass_outputs = model(inputs)

            # Convert outputs to predictions
            binary_preds = (binary_outputs > 0.5).int().cpu().numpy()
            multiclass_preds = torch.argmax(multiclass_outputs, dim=1).cpu().numpy()

            # Store predictions and targets
            all_binary_preds.extend(binary_preds.flatten())
            all_multiclass_preds.extend(multiclass_preds)
            all_binary_targets.extend(binary_targets.numpy())
            all_multiclass_targets.extend(multiclass_targets.numpy())

    # Convert lists to numpy arrays
    all_binary_preds = np.array(all_binary_preds)
    all_multiclass_preds = np.array(all_multiclass_preds)
    all_binary_targets = np.array(all_binary_targets)
    all_multiclass_targets = np.array(all_multiclass_targets)

    # Calculate metrics for binary classification
    binary_accuracy = accuracy_score(all_binary_targets, all_binary_preds)
    binary_precision = precision_score(all_binary_targets, all_binary_preds)
    binary_recall = recall_score(all_binary_targets, all_binary_preds)
    binary_f1 = f1_score(all_binary_targets, all_binary_preds)
    binary_confusion_matrix = confusion_matrix(all_binary_targets, all_binary_preds)

    # Calculate metrics for multi-class classification
    multi_class_accuracy = accuracy_score(all_multiclass_targets, all_multiclass_preds)
    multi_class_precision = precision_score(all_multiclass_targets, all_multiclass_preds, average='weighted')
    multi_class_recall = recall_score(all_multiclass_targets, all_multiclass_preds, average='weighted')
    multi_class_f1 = f1_score(all_multiclass_targets, all_multiclass_preds, average='weighted')
    multi_class_confusion_matrix = confusion_matrix(all_multiclass_targets, all_multiclass_preds)

    # Print binary classification metrics
    print(f'Binary Classification: ')
    print(f'Accuracy: {binary_accuracy}')
    print(f'Precision: {binary_precision}')
    print(f'Recall: {binary_recall}')
    print(f'F1 Score: {binary_f1}')
    print(f'Confusion Matrix: \n{binary_confusion_matrix}')

    # Print multi-class classification metrics
    print(f'\nMulti-Class Classification: ')
    print(f'Accuracy: {multi_class_accuracy}')
    print(f'Precision: {multi_class_precision}')
    print(f'Recall: {multi_class_recall}')
    print(f'F1 Score: {multi_class_f1}')
    print(f'Confusion Matrix: \n{multi_class_confusion_matrix}')

    # Store the metrics
    metrics = {
        'Binary': {
            'Accuracy': binary_accuracy,
            'Precision': binary_precision,
            'Recall': binary_recall,
            'F1 Score': binary_f1
        },
        'Multi-Class': {
            'Accuracy': multi_class_accuracy,
            'Precision': multi_class_precision,
            'Recall': multi_class_recall,
            'F1 Score': multi_class_f1
        }
    }

    return metrics, binary_confusion_matrix, multi_class_confusion_matrix


def plot_metrics(metrics):
    """Plot the evaluation metrics"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    binary_values = [metrics['Binary'][cat] for cat in categories]
    multi_class_values = [metrics['Multi-Class'][cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, binary_values, width, label='Binary Classification', color='#003566')
    rects2 = ax.bar(x + width / 2, multi_class_values, width, label='Multi-Class Classification', color='#ffc300')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('CNN + Bi-GRU with Selected Features (PyTorch)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='lower right')

    # Function to add labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 4)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()


def main():
    # Set paths to your data files
    train_path = '../../dataset/UNSW_NB15_testing-set.csv'
    test_path = '../../dataset/UNSW_NB15_testing-set.csv'

    # Create model checkpoint directory
    checkpoint_dir = "model_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load and preprocess data
    X_train, X_test, y_train_binary, y_test_binary, y_train_multi_class, y_test_multi_class = load_data(train_path,
                                                                                                        test_path)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_binary_tensor = torch.FloatTensor(y_train_binary.values)
    y_train_multi_class_tensor = torch.LongTensor(y_train_multi_class.values)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_binary_tensor = torch.FloatTensor(y_test_binary.values)
    y_test_multi_class_tensor = torch.LongTensor(y_test_multi_class.values)

    # Create PyTorch datasets and dataloaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_binary_tensor, y_train_multi_class_tensor)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(X_test_tensor, y_test_binary_tensor, y_test_multi_class_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Count number of unique classes in multi-class classification
    num_classes = len(np.unique(y_train_multi_class))
    print(f"Number of unique classes: {num_classes}")

    # Initialize model
    input_dim = X_train.shape[1]
    hidden_dim = 50
    model = CNNBiGRUModel(input_dim, hidden_dim, num_classes, dropout=0.01).to(device)
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
        'num_classes': num_classes
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Plot training history
    plt.figure(figsize=(12, 8))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'], label='Training Loss')
    plt.plot(history['val_total_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot binary accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['binary_acc'], label='Training Accuracy')
    plt.plot(history['val_binary_acc'], label='Validation Accuracy')
    plt.title('Binary Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot multiclass accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['multiclass_acc'], label='Training Accuracy')
    plt.plot(history['val_multiclass_acc'], label='Validation Accuracy')
    plt.title('Multiclass Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))
    plt.show()

    # Evaluate model on test set
    # Load the best model for evaluation
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} for evaluation")

    # Evaluate model
    metrics, binary_cm, multi_class_cm = evaluate_model(model, test_loader)

    # Plot metrics
    plot_metrics(metrics)

    # Save evaluation results
    import json
    with open(os.path.join(checkpoint_dir, 'evaluation_metrics.json'), 'w') as F:
        json.dump({
            'Binary': {k: float(v) for k, v in metrics['Binary'].items()},
            'Multi-Class': {k: float(v) for k, v in metrics['Multi-Class'].items()}
        }, F, indent=4)


if __name__ == "__main__":
    main()
