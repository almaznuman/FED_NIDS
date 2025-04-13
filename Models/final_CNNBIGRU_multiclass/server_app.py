# import logging
from typing import List, Tuple, Optional, Dict

import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from Models.final_CNNBIGRU_multiclass.custom_fed_avg import CustomFedAvg
from Models.final_CNNBIGRU_multiclass.loss_strat import ReliabilityIndex
from Models.final_CNNBIGRU_multiclass.task import load_model, load_cen_data, get_weights, set_weights, test


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics weighted by number of examples.

    Args:
        metrics: List of tuples (num_examples, metrics_dict)

    Returns:
        Aggregated metrics dictionary
    """
    # Weighted sums
    total_examples = sum(num_examples for num_examples, _ in metrics)
    loss_sum, acc_sum, prec_sum, rec_sum, f1_sum = 0.0, 0.0, 0.0, 0.0, 0.0

    for (num_examples, m) in metrics:
        if "loss" in m:
            loss_sum += m["loss"] * num_examples
        acc_sum += m["accuracy"] * num_examples
        prec_sum += m["precision"] * num_examples
        rec_sum += m["recall"] * num_examples
        f1_sum += m["f1"] * num_examples

    # Compute weighted averages
    aggregated_metrics = {
        "accuracy": acc_sum / total_examples,
        "precision": prec_sum / total_examples,
        "recall": rec_sum / total_examples,
        "f1": f1_sum / total_examples,
    }
    
    # Add loss if it exists in any of the metrics
    if loss_sum > 0:
        aggregated_metrics["loss"] = loss_sum / total_examples
        
    return aggregated_metrics


def get_evaluate_fn():
    """Return an evaluation function for server-side centralized evaluation.

    Returns:
        Function that evaluates the global model on the central test set
    """
    # Load the centralized test dataset
    testloader = load_cen_data()  # Returns DataLoader

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluation function used during training rounds
    def evaluate(
            server_round: int,
            parameters: ndarrays_to_parameters,
            config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model on the centralized test dataset.

        Args:
            server_round: Current round of federated learning
            parameters: Model parameters to evaluate
            config: Configuration parameters

        Returns:
            Tuple of (loss, metrics dictionary)
        """
        # Load model and criterion with default num_classes if not in config
        model_dict = load_model(num_classes=10)
        model = model_dict["model"]
        criterion = model_dict["criterion"]

        # Set model parameters
        set_weights(model, parameters)

        # Evaluate the model
        loss, y_pred, y_true = test(model, testloader, criterion, device)

        # Calculate metrics for multiclass classification
        accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
        precision = precision_score(y_true.numpy(), y_pred.numpy(), average='weighted')
        recall = recall_score(y_true.numpy(), y_pred.numpy(), average='weighted')
        f1 = f1_score(y_true.numpy(), y_pred.numpy(), average='weighted')

        # Return loss and performance metrics
        return loss, {
            "cen_accuracy": float(accuracy),
            "cen_precision": float(precision),
            "cen_recall": float(recall),
            "cen_f1": float(f1),
        }

    return evaluate


def server_fn(context: Context):
    """Construct server with trust-based strategy for federated learning.

    Args:
        context: Server context

    Returns:
        ServerAppComponents with trust-based strategy and configuration
    """
    # Initialize model to extract weights for initial parameters
    model_dict = load_model(num_classes=10)
    model = model_dict["model"]

    # Get model parameters
    # fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    min_available_clients = context.run_config["min-available-clients"]
    min_evaluate_clients = context.run_config["min-evaluate-clients"]
    
    # Get strategy type from run config with default fallback to "diversity"
    strategy_type = context.run_config["strategy-type"]

    # Get model parameters
    parameters = ndarrays_to_parameters(get_weights(model))

    # Select strategy based on strategy_type
    if strategy_type == "reliability_index":
        diversity_weight = context.run_config["diversity-weight"]
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
            evaluate_fn=get_evaluate_fn(),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:  # fedavg or any other value defaults to CustomFedAvg
        strategy = CustomFedAvg(
            run_config=context.run_config,
            use_wandb=context.run_config["use-wandb"],
            fraction_fit=0.3,
            min_fit_clients=3,
            fraction_evaluate=fraction_eval,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=parameters,
            evaluate_fn=get_evaluate_fn(),
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    # Create server configuration
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp with trust-based strategy
app = ServerApp(server_fn=server_fn)