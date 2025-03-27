"""pytorch-example: A Flower / PyTorch app with FedProx implementation."""

import json
import copy
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from flwr.common.typing import UserConfig

FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
    [
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)


class Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(net, trainloader, epochs, lr, device, proximal_mu=0.0, global_parameters=None):
    """Train the model on the training set with proximal term regularization (FedProx).
    
    Args:
        net: The local model
        trainloader: DataLoader for training data
        epochs: Number of local training epochs
        lr: Learning rate
        device: Training device (CPU/GPU)
        proximal_mu: The weight of the proximal term (0 = standard SGD)
        global_parameters: The global model parameters to use for proximal regularization
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
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
            images = batch["image"]
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def apply_train_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
    return batch


def apply_eval_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [EVAL_TRANSFORMS(img) for img in batch["image"]]
    return batch


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.1,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_test["train"].with_transform(
        apply_train_transforms
    )
    test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader


def create_run_dir(config: UserConfig) -> Path:
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