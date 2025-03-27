"""CIFAR-10 with FedProx implementation for PyTorch."""

import copy
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import f1_score, precision_score, recall_score

from flwr.common.typing import UserConfig


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Use DirichletPartitioner for non-IID data
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,  # Lower alpha creates more non-IID partitioning
            seed=42,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device, lr=0.01, proximal_mu=0.0, global_parameters=None):
    """Train the model on the training set with proximal term regularization (FedProx).
    
    Args:
        net: The local model
        trainloader: DataLoader for training data
        epochs: Number of local training epochs
        device: Training device (CPU/GPU)
        lr: Learning rate
        proximal_mu: The weight of the proximal term (0 = standard training)
        global_parameters: The global model parameters to use for proximal regularization
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Initialize global_net for proximal term if proximal_mu > 0
    global_net = None
    if proximal_mu > 0 and global_parameters is not None:
        global_net = copy.deepcopy(net)
        set_weights(global_net, global_parameters)
        global_net.to(device)
    
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss = criterion(net(images.to(device)), labels.to(device))
            
            # Add proximal term if FedProx is enabled
            if proximal_mu > 0 and global_net is not None:
                proximal_term = 0.0
                # Calculate L2 norm between local and global model parameters
                for local_param, global_param in zip(net.parameters(), global_net.parameters()):
                    proximal_term += torch.norm(local_param - global_param) ** 2
                
                # Add the proximal term to the loss
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predictions = torch.max(outputs.data, 1)[1]
            correct += (predictions == labels).sum().item()

            # Store labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate metrics
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    # Calculate additional metrics using scikit-learn
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return loss, accuracy, f1, precision, recall


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def apply_eval_transforms(batch):
    """Apply transforms to evaluation data."""
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir 