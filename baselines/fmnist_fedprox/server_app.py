"""FedProx server implementation for Fashion MNIST."""

import torch

from baselines.fmnist_fedprox.strategy import CustomFedProx
from baselines.fmnist_fedprox.task import (
    Net,
    apply_eval_transforms,
    get_weights,
    set_weights,
    test,
)
from torch.utils.data import DataLoader

from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """Return training configuration for each round."""
    # Adjust learning rate based on round
    lr = 0.01
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):
    """Compute weighted average of metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Return server components for federated learning."""
    # Hardcoded configuration values
    num_rounds = 10
    fraction_fit = 0.3
    fraction_eval = 1
    min_available_clients = 10
    min_fit_clients = 3
    min_evaluate_clients = 10
    server_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    proximal_mu = 0.2  # Ideal value for Fashion MNIST
    use_wandb = False

    # Store values in run_config for consistency
    # context.run_config["num-server-rounds"] = num_rounds
    # context.run_config["fraction-fit"] = fraction_fit
    # context.run_config["fraction-evaluate"] = fraction_eval
    # context.run_config["min-available-clients"] = min_available_clients
    # context.run_config["min-fit-clients"] = min_fit_clients
    # context.run_config["min-evaluate-clients"] = min_evaluate_clients
    # context.run_config["server-device"] = server_device
    # context.run_config["proximal-mu"] = proximal_mu
    # context.run_config["use-wandb"] = use_wandb

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation
    global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]
    testloader = DataLoader(
        global_test_set.with_transform(apply_eval_transforms),
        batch_size=32,
    )

    # Initialize the FedProx strategy
    strategy = CustomFedProx(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        fraction_evaluate=fraction_eval,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=proximal_mu,  # The proximal term coefficient
    )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn) 