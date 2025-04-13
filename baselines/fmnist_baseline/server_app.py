"""pytorch-example: A Flower / PyTorch app."""

import torch

from baselines.fmnist_baseline.strategy import CustomFedAvg
from baselines.fmnist_baseline.task import (
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

from baselines.fmnist_baseline.loss_Strat import ReliabilityIndex

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
        loss, accuracy, f1, precision, recall = test(net, testloader, device=device)
        return loss, {
            "centralized_accuracy": accuracy,
            "centralized_f1": f1,
            "centralized_precision": precision,
            "centralized_recall": recall
        }

    return evaluate


def on_fit_config(server_round: int):
    lr = 0.01  # reduced learning rate
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate additional metrics (weighted average)
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]

    # Aggregate and return all metrics (weighted average)
    return {
        "federated_evaluate_accuracy": sum(accuracies) / sum(examples),
        "federated_evaluate_f1": sum(f1_scores) / sum(examples),
        "federated_evaluate_precision": sum(precisions) / sum(examples),
        "federated_evaluate_recall": sum(recalls) / sum(examples)
    }


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    # fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    min_available_clients = context.run_config["min-available-clients"]
    min_evaluate_clients = context.run_config["min-evaluate-clients"]
    server_device = context.run_config["server-device"]
    diversity_weight = context.run_config["diversity-weight"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation
    global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]

    testloader = DataLoader(
        global_test_set.with_transform(apply_eval_transforms),
        batch_size=32,
    )

    strategy_type = context.run_config["strategy-type"]
    alpha = context.run_config["alpha"]

    if strategy_type == "reliability_index":

        # Initialize the diversity-aware strategy
        strategy = ReliabilityIndex(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            fraction_fit=1,
            min_fit_clients=10,
            num_clients_to_select=3,
            diversity_weight=diversity_weight,
            fraction_evaluate=fraction_eval,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
            evaluate_metrics_aggregation_fn=weighted_average,
            strategy_type=strategy_type,
            alpha=alpha
        )
    else:
        # Define strategy
        strategy = CustomFedAvg(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            fraction_fit=0.3,
            min_fit_clients=3,
            fraction_evaluate=fraction_eval,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
            evaluate_metrics_aggregation_fn=weighted_average,
            strategy_type=strategy_type,
            alpha=alpha
        )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
