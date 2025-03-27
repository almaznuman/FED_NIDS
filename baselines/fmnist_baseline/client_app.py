"""pytorch-example: A Flower / PyTorch app."""

import torch
from baselines.fmnist_baseline.task import Net, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, RecordDict


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
        self, net, client_state: RecordDict, trainloader, valloader, local_epochs
    ):
        self.net: Net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # Apply weights from global models (the whole model is replaced)
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            lr=float(config["lr"]),
            device=self.device,
        )
        # Save classification head to context's state to use in a future fit() call
        self._save_layer_weights_to_state()

        # Calculate model similarity/diversity metric
        diversity_score = self._calculate_model_diversity(parameters)

        self._verify_state()

        # Return locally-trained model and metrics
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "diversity_score": diversity_score},
        )

    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        arr_record = ArrayRecord(self.net.fc2.state_dict())

        # Add to RecordDict (replace if already exists)
        self.client_state[self.local_layer_name] = arr_record

    def _verify_state(self):
        if self.local_layer_name in self.client_state.array_records:
            # Retrieve the ArrayRecord object
            arr_record = self.client_state[self.local_layer_name]
            # Convert it to a torch state dict for inspection
            stored_state = arr_record.to_torch_state_dict()
            # Print the keys and shapes of the stored parameters
            print("Verified stored layer weights for key '{}':".format(self.local_layer_name))
            for name, param in stored_state.items():
                print("  {}: shape {}".format(name, param.shape))
        else:
            print("No stored weights found for key '{}' in client state.".format(self.local_layer_name))

    def _calculate_model_diversity(self, global_parameters):
        """Calculate how different this client's model is from the global model.
        
        Returns a score where higher values indicate greater diversity.
        """
        # If we don't have personalized weights yet, return default value
        if self.local_layer_name not in self.client_state.array_records:
            return 0.0
        
        # Get the global classification layer's parameters
        global_model = Net()
        set_weights(global_model, global_parameters)
        global_fc2_params = {name: param for name, param in global_model.fc2.named_parameters()}
        
        # Get our personalized layer parameters
        personalized_layer = self.client_state[self.local_layer_name]
        personalized_state = personalized_layer.to_torch_state_dict()
        
        # Calculate difference between weights (L2 norm)
        total_diff = 0.0
        total_weight = 0
        
        for name, local_param in personalized_state.items():
            if name in global_fc2_params:
                param_diff = torch.norm(local_param - global_fc2_params[name]).item()
                param_size = local_param.numel()
                total_diff += param_diff
                total_weight += param_size
        
        # Normalize by parameter count
        if total_weight > 0:
            avg_diff = total_diff / total_weight
        else:
            avg_diff = 0.0
        
        return avg_diff

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set.

        Calculates the L2 norm between the global model and the personalized model,
        then returns this as an additional metric.
        """
        set_weights(self.net, parameters)

        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state
    return FlowerClient(
        net, client_state, trainloader, valloader, local_epochs
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
