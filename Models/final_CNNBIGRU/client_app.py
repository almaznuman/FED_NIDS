"""BiGRU: A Flower / PyTorch app for anomaly detection."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ArrayRecord, RecordDict
from Models.final_CNNBIGRU.task import (
    load_data, 
    load_model, 
    get_weights, 
    set_weights, 
    train, 
    test
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch


class FlowerClient(NumPyClient):
    """Flower client implementing anomaly detection with BiGRU."""

    def __init__(
            self,
            learning_rate,
            trainloader,
            testloader,
            epochs,
            batch_size,
            verbose,
            client_state: RecordDict,
    ):
        """Initialize the client with model and data.
        
        Args:
            learning_rate: Learning rate for optimizer
            trainloader: DataLoader for training data
            testloader: DataLoader for test data
            epochs: Number of local epochs
            batch_size: Batch size for training
            verbose: Whether to print verbose output
            client_state: Client state for persisting data between rounds
        """
        # Initialize model and optimizer
        model_dict = load_model(learning_rate)
        self.model = model_dict["model"]
        self.optimizer = model_dict["optimizer"]
        self.criterion = model_dict["criterion"]
        
        # Store data loaders
        self.trainloader = trainloader
        self.testloader = testloader
        
        # Store training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Store client state for personalization
        self.client_state = client_state
        self.local_layer_name = "classification-head"

    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        arr_record = ArrayRecord(self.model.fc.state_dict())
        self.client_state[self.local_layer_name] = arr_record

    def _verify_state(self):
        """Verify that weights were saved correctly."""
        if self.local_layer_name in self.client_state.array_records:
            arr_record = self.client_state[self.local_layer_name]
            stored_state = arr_record.to_torch_state_dict()
            if self.verbose:
                print(f"Verified stored layer weights for key '{self.local_layer_name}':")
                for name, param in stored_state.items():
                    print(f"  {name}: shape {param.shape}")
        else:
            if self.verbose:
                print(f"No stored weights found for key '{self.local_layer_name}' in client state.")

    def _calculate_model_diversity(self, global_parameters):
        if self.local_layer_name not in self.client_state.array_records:
            return 0.0
        
        # Get the global classification layer's parameters
        global_model_dict = load_model()
        global_model = global_model_dict["model"]
        set_weights(global_model, global_parameters)
        global_fc_params = {name: param for name, param in global_model.fc.named_parameters()}
        
        # Get our personalized layer parameters
        personalized_layer = self.client_state[self.local_layer_name]
        personalized_state = personalized_layer.to_torch_state_dict()
        
        # Calculate difference between weights (L2 norm)
        total_diff = 0.0
        total_weight = 0
        
        for name, local_param in personalized_state.items():
            if name in global_fc_params:
                param_diff = torch.norm(local_param - global_fc_params[name]).item()
                param_size = local_param.numel()
                total_diff += param_diff
                total_weight += param_size
        
        # Normalize by parameter count
        if total_weight > 0:
            avg_diff = total_diff / total_weight
        else:
            avg_diff = 0.0
        
        return avg_diff

    def fit(self, parameters, config):
        """Train the model with data of this client.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration from the server
            
        Returns:
            tuple: (updated_parameters, num_examples, metrics)
        """
        # Update model parameters
        set_weights(self.model, parameters)
        
        # Train the model
        train_loss = train(
            self.model,
            self.trainloader,
            self.epochs,
            self.optimizer,
            self.criterion,
            self.device
        )
        
        # Save classification head to state for future use
        self._save_layer_weights_to_state()
        
        # Calculate model diversity score
        diversity_score = self._calculate_model_diversity(parameters)
        
        # Verify state was saved correctly
        self._verify_state()
        
        # Return updated model parameters and metrics
        return get_weights(self.model), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "diversity_score": diversity_score
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration from the server
            
        Returns:
            tuple: (loss, num_examples, metrics)
        """
        # Update model parameters
        set_weights(self.model, parameters)
        
        # Evaluate model
        loss, y_pred, y_test = test(self.model, self.testloader, self.criterion, self.device)
        
        # Calculate metrics
        y_pred_binary = (y_pred > 0.5).int().numpy()
        y_test_numpy = y_test.numpy()
            
        accuracy = accuracy_score(y_test_numpy, y_pred_binary)
        precision = precision_score(y_test_numpy, y_pred_binary)
        recall = recall_score(y_test_numpy, y_pred_binary)
        f1 = f1_score(y_test_numpy, y_pred_binary)
        
        return loss, len(self.testloader.dataset), {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    

def client_fn(context: Context):
    """Construct a client for the Flower simulation.
    
    Args:
        context: Flower client context
        
    Returns:
        A Flower client
    """
    # Get client config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load data
    trainloader, testloader = load_data(partition_id, num_partitions)

    # Get run config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config["verbose"]
    learning_rate = context.run_config["learning-rate"]
    
    # Get client state for personalization
    client_state = context.state

    # Create and return client
    return FlowerClient(
        learning_rate, 
        trainloader, 
        testloader, 
        epochs, 
        batch_size, 
        verbose,
        client_state
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
