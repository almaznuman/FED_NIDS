"""FedProx client implementation for Fashion MNIST."""

import torch
from baselines.prox.fmnist_fedprox.task import Net, get_weights, load_data, set_weights, train, test

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


# Define Flower Client
class FlowerClient(NumPyClient):
    """A client implementation supporting FedProx.
    
    This client uses the proximal term for regularization during training.
    """

    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Train model with proximal regularization.
        
        The FedProx proximal_mu parameter is passed from the server 
        via the config dictionary.
        """
        # Get proximal_mu from config (default to 0 if not provided)
        proximal_mu = config.get("proximal_mu", 0.0)
        
        # Get learning rate 
        lr = config.get("lr", 0.01)
        
        # Apply global model parameters
        set_weights(self.net, parameters)
        
        # Train with proximal regularization
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            lr=float(lr),
            device=self.device,
            proximal_mu=proximal_mu,  # Pass the proximal term coefficient
            global_parameters=parameters,  # Pass global model parameters
        )

        # Return the trained model
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set."""
        set_weights(self.net, parameters)
        
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Create and return a Flower client."""
    # Load model
    net = Net()
    
    # Get partition information
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load data for this partition
    trainloader, valloader = load_data(partition_id, num_partitions)
    
    # Get local training config
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Create Flower ClientApp
app = ClientApp(
    client_fn,
) 