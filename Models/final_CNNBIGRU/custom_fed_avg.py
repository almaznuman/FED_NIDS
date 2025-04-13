"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO

import torch
import wandb
from Models.final_CNNBIGRU.task import load_model, create_run_dir, set_weights
from flwr.common import (
    Parameters,
)
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

PROJECT_NAME = "DEMO UNSW Binary"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, strategy_type: str, alpha: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        self.strategy_type = strategy_type
        self.alpha = alpha
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        """Initialize Weights & Biases project for logging."""
        wandb.init(project=PROJECT_NAME, name=f"{self.strategy_type}- Alpha {self.alpha}")

    def _store_results(self, tag: str, results_dict: dict) -> None:
        """Store results in an internal dictionary and write them to disk as JSON.
        
        Args:
            tag: Category tag for the results
            results_dict: Dictionary of results to store
        """
        # Add results to the appropriate category
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]
        
        # Write results to disk
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def store_results_and_log(self, server_round: int, tag: str, results_dict: dict) -> None:
        """Store results and optionally log them to W&B.
        
        Args:
            server_round: Current server round
            tag: Category tag for the results
            results_dict: Dictionary of results to store and log
        """
        # Add the round number to the results and store them
        results_with_round = {"round": server_round, **results_dict}
        self._store_results(tag, results_with_round)
        if self.use_wandb:
            wandb.log(results_dict, step=server_round)

    def _update_best_acc(self, server_round: int, accuracy: float, parameters: Parameters) -> None:
        """If a new best accuracy is achieved, checkpoint the model.
        
        Args:
            server_round: Current server round
            accuracy: Accuracy achieved in this round
            parameters: Model parameters to save if they're the best so far
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "New best global model found: %f", accuracy)
            
            # Convert parameters to ndarrays and update PyTorch model
            ndarrays = parameters_to_ndarrays(parameters)
            model_dict = load_model()
            model = model_dict["model"]
            
            # Set model parameters
            set_weights(model, ndarrays)
            
            # Save model checkpoint
            file_name = self.save_path / f"model_state_acc_{accuracy:.3f}_round_{server_round}.pt"
            torch.save(model.state_dict(), file_name)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["cen_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
