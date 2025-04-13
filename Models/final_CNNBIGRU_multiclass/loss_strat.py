"""Client selection strategies combining loss-based and gradient similarity approaches for CNNBIGRU."""

import json
import random
from logging import INFO, WARNING

import torch
import wandb
from flwr.common import (
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    logger,
    parameters_to_ndarrays,
)
from flwr.common.typing import UserConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from Models.final_CNNBIGRU_multiclass.task import create_run_dir, load_model, set_weights

# Define a project name for logging (if using W&B)
PROJECT_NAME = "DEMO UNSW Multi"


class LossBasedSelectionStrategy(FedAvg):
    """
    Strategy that selects clients solely based on their training loss values.
    Clients with lower loss values are preferred.
    """

    def __init__(
            self,
            run_config: UserConfig,
            use_wandb: bool,
            # Selection parameters
            num_clients_to_select: int = 10,
            # FedAvg parameters
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: callable = None,
            on_fit_config_fn: callable = None,
            on_evaluate_config_fn: callable = None,
            accept_failures: bool = True,
            initial_parameters: Parameters = None,
            fit_metrics_aggregation_fn: MetricsAggregationFn = None,
            evaluate_metrics_aggregation_fn: MetricsAggregationFn = None,
            inplace: bool = True,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )
        # Client selection parameters
        self.num_clients_to_select = num_clients_to_select
        self.client_loss_history = {}

        # Setup for logging and result tracking
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        if self.use_wandb:
            self._init_wandb_project()

        self.best_acc_so_far = 0.0
        self.results = {}

    def _init_wandb_project(self):
        """Initialize Weights & Biases logging."""
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round_number, accuracy, parameters):
        """Determines if a new best global model has been found."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "New best global model found: %f", accuracy)
            ndarrays = parameters_to_ndarrays(parameters)
            model_dict = load_model()
            model = model_dict["model"]
            set_weights(model, ndarrays)
            file_name = f"model_state_acc_{accuracy}_round_{round_number}.pt"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            wandb.log(results_dict, step=server_round)

    def update_client_loss_history(self, results: list[tuple[ClientProxy, FitRes]]):
        """Update client loss history using training results."""
        for client, fit_res in results:
            if fit_res.metrics and "train_loss" in fit_res.metrics:
                current_loss = fit_res.metrics["train_loss"]
                # Normalize loss by the number of samples
                num_samples = fit_res.num_examples
                normalized_loss = current_loss / num_samples if num_samples > 0 else float('inf')
                self.client_loss_history[client.cid] = normalized_loss
                logger.log(INFO,
                           f"Client {client.cid}: raw loss={current_loss}, samples={num_samples}, normalized={normalized_loss}")

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes]] | list[BaseException],
            skip_selection: bool = False
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results and select clients based on loss."""
        if not results:
            return None, {}

        # Update client loss history from all results
        self.update_client_loss_history(results)

        # Skip selection if requested (for child classes that do their own selection)
        if skip_selection:
            return super().aggregate_fit(server_round, results, failures)

        # Random selection for first round when loss history is not available
        # if server_round == 1:
        #     selected_results = random.sample(results, min(self.num_clients_to_select, len(results)))
        #     logger.log(WARNING,
        #                f"Round {server_round}: Random Selection Used - Clients: {[client.cid for client, _ in selected_results]}")
        # else:
        #     # Loss-based selection
        #     client_losses = []
        #     for client, fit_res in results:
        #         loss = self.client_loss_history.get(client.cid, float('inf'))
        #         client_losses.append((client, fit_res, loss))
        #
        #     # Sort by loss (ascending - lower is better)
        #     client_losses.sort(key=lambda x: x[2])
        #
        #     # Log client losses
        #     logger.log(WARNING,
        #                f"Round {server_round}: Client losses: {[(c.cid, loss) for c, _, loss in client_losses]}")
        #
        #     # Select top clients with lowest loss
        #     selected_results = [(client, fit_res) for client, fit_res, _ in
        #                         client_losses[:min(self.num_clients_to_select, len(client_losses))]]
        #
        #     logger.log(WARNING,
        #                f"Round {server_round}: Selected {len(selected_results)} clients based on loss: {[c.cid for c, _ in selected_results]}")

        # Call parent's aggregate_fit with selected clients only

        # Loss-based selection
        client_losses = []
        for client, fit_res in results:
            loss = self.client_loss_history.get(client.cid, float('inf'))
            client_losses.append((client, fit_res, loss))

        # Sort by loss (ascending - lower is better)
        client_losses.sort(key=lambda x: x[2])

        # Log client losses
        logger.log(WARNING,
                   f"Round {server_round}: Client losses: {[(c.cid, loss) for c, _, loss in client_losses]}")

        # Select top clients with lowest loss
        selected_results = [(client, fit_res) for client, fit_res, _ in
                            client_losses[:min(self.num_clients_to_select, len(client_losses))]]

        logger.log(WARNING,
                   f"Round {server_round}: Selected {len(selected_results)} clients based on loss: {[c.cid for c, _ in selected_results]}")

        parameters, metrics = super().aggregate_fit(server_round, selected_results, failures)

        return parameters, metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[tuple[ClientProxy, EvaluateRes]] | list[BaseException]
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation results."""
        # Call the base aggregation
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Log evaluation results
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics}
        )
        return loss, metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, dict[str, Scalar]]:
        """Evaluate the current global model."""
        # Call the base evaluation
        loss, metrics = super().evaluate(server_round, parameters)

        # Check for new best accuracy
        # For CNNBIGRU, we might prefer to use 'cen_f1' or 'cen_auc' instead of accuracy
        if 'cen_f1' in metrics:
            acc = metrics.get("cen_f1", 0.0)
        elif 'cen_accuracy' in metrics:
            acc = metrics.get("cen_accuracy", 0.0)
        else:
            acc = 0.0
            
        self._update_best_acc(server_round, acc, parameters)

        # Log centralized evaluation metrics
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics}
        )
        return loss, metrics

class ReliabilityIndex(LossBasedSelectionStrategy):
    """
    Strategy that selects clients based on both loss and model diversity.
    Balances exploration (diversity) and exploitation (performance).
    """

    def __init__(
            self,
            run_config: UserConfig,
            use_wandb: bool,
            # Selection parameters
            num_clients_to_select: int = 10,
            diversity_weight: float = 0.3,  # How much to value diversity vs loss
            strategy_type: str = "base",
            alpha: float = 0.0,
            # FedAvg parameters
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: callable = None,
            on_fit_config_fn: callable = None,
            on_evaluate_config_fn: callable = None,
            accept_failures: bool = True,
            initial_parameters: Parameters = None,
            fit_metrics_aggregation_fn: MetricsAggregationFn = None,
            evaluate_metrics_aggregation_fn: MetricsAggregationFn = None,
            inplace: bool = True,
    ) -> None:
        # Set diversity_weight before calling super().__init__
        self.diversity_weight = diversity_weight
        self.client_diversity_history = {}
        self.strategy_type = strategy_type
        self.alpha = alpha
        
        super().__init__(
            run_config=run_config,
            use_wandb=use_wandb,
            num_clients_to_select=num_clients_to_select,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )

    def _init_wandb_project(self):
        """Initialize Weights & Biases logging with diversity weight in run name."""
        # wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-DW{self.diversity_weight}-ServerApp")
        wandb.init(project=PROJECT_NAME, name=f"{self.strategy_type}, Alpha {self.alpha}")

    def update_client_diversity_history(self, results: list[tuple[ClientProxy, FitRes]]):
        """Update client diversity history using training results."""
        for client, fit_res in results:
            if fit_res.metrics and "diversity_score" in fit_res.metrics:
                diversity_score = fit_res.metrics["diversity_score"]
                self.client_diversity_history[client.cid] = diversity_score
                logger.log(INFO, f"Client {client.cid}: diversity score={diversity_score}")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes]] | list[BaseException], **kwargs) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results with diversity-aware selection."""
        if not results:
            return None, {}

        # Update client metrics history
        self.update_client_loss_history(results)
        self.update_client_diversity_history(results)

        # Random selection for first round when history is incomplete
        # if server_round == 1:
        #     selected_results = random.sample(results, min(self.num_clients_to_select, len(results)))
        #     logger.log(WARNING,
        #                f"Round {server_round}: Random Selection Used - Clients: {[client.cid for client, _ in selected_results]}")
        # else:
        #     # Calculate combined scores (lower is better for loss, higher is better for diversity)
        #     client_scores = []
        #     for client, fit_res in results:
        #         loss = self.client_loss_history.get(client.cid, float('inf'))
        #         diversity = self.client_diversity_history.get(client.cid, 0.0)
        #
        #         # Normalize and combine scores
        #         # -1 * diversity because higher diversity is better, but we sort ascending
        #         combined_score = (1 - self.diversity_weight) * loss - self.diversity_weight * diversity
        #
        #         client_scores.append((client, fit_res, combined_score, loss, diversity))
        #
        #     # Sort by combined score (lower is better)
        #     client_scores.sort(key=lambda x: x[2])
        #
        #     # Log client scores
        #     logger.log(WARNING,
        #                f"Round {server_round}: Client scores: {[(c.cid, score, loss, div) for c, _, score, loss, div in client_scores]}")
        #
        #     # Select top clients with best combined scores
        #     selected_results = [(client, fit_res) for client, fit_res, _, _, _ in
        #                         client_scores[:min(self.num_clients_to_select, len(client_scores))]]
        #
        #     logger.log(WARNING,
        #                f"Round {server_round}: Selected {len(selected_results)} clients based on diversity-aware score")

        # Call parent's aggregate_fit with selected clients only, skipping selection

        # Calculate combined scores (lower is better for loss, higher is better for diversity)
        client_scores = []
        for client, fit_res in results:
            loss = self.client_loss_history.get(client.cid, float('inf'))
            diversity = self.client_diversity_history.get(client.cid, 0.0)

            # Normalize and combine scores
            # -1 * diversity because higher diversity is better, but we sort ascending
            combined_score = (1 - self.diversity_weight) * loss - self.diversity_weight * diversity

            client_scores.append((client, fit_res, combined_score, loss, diversity))

        # Sort by combined score (lower is better)
        client_scores.sort(key=lambda x: x[2])

        # Log client scores
        logger.log(WARNING,
                   f"Round {server_round}: Client scores: {[(c.cid, score, loss, div) for c, _, score, loss, div in client_scores]}")

        # Select top clients with best combined scores
        selected_results = [(client, fit_res) for client, fit_res, _, _, _ in
                            client_scores[:min(self.num_clients_to_select, len(client_scores))]]

        logger.log(WARNING,
                   f"Round {server_round}: Selected {len(selected_results)} clients based on Reliability Index")

        parameters, metrics = super().aggregate_fit(server_round, selected_results, failures, skip_selection=True)

        return parameters, metrics 