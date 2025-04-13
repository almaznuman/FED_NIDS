# FED-NIDS

A federated learning-based network intrusion detection system (NIDS) using PyTorch and Flower framework with the UNSW-NB15 dataset.

## 📋 Overview

This project implements a federated learning approach to network intrusion detection, providing a privacy-preserving way to train models across multiple decentralized edge devices without sharing raw data. Using advanced deep learning architectures (CNN-BiGRU), it achieves high accuracy in detecting various network attacks.

### Key Features

- 🔒 Privacy-preserving federated learning for intrusion detection
- 🧠 Advanced CNN-BiGRU models for binary and multiclass classification
- 📊 Integration with Weights & Biases for experiment tracking
- 🌐 Customizable federated learning strategies including FedAvg and Diversity-aware Aggregation
- 🖥️ User-friendly Streamlit interface for running experiments

## 🛠️ Architecture

The project is built with:

- **PyTorch**: Deep learning framework for implementing neural networks
- **Flower**: Federated Learning framework for distributed training
- **Streamlit**: Frontend UI for experiment configuration and execution
- **Weights & Biases**: Optional integration for experiment tracking

### Models

Two main model architectures are implemented:

1. **CNN-BiGRU (Binary)**: A hybrid model combining Convolutional Neural Networks and Bidirectional Gated Recurrent Units for binary classification (normal/attack)
2. **CNN-BiGRU (Multiclass)**: Extended architecture for multiclass classification of different attack types

## 📊 Dataset

This project uses the UNSW-NB15 dataset, a comprehensive network traffic dataset containing normal activities and various synthetic attack behaviors. The dataset needs to be downloaded separately from:
https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS

After downloading, place the following files in the `dataset` directory:
- UNSW_NB15_training-set.csv
- UNSW_NB15_testing-set.csv

## 🧪 Baseline Experiments

The project includes several baseline federated learning experiments to benchmark performance:

### CIFAR-10 Baseline
- Standard CNN architecture for image classification
- Configurable number of clients and federated rounds

### Fashion MNIST Baseline
- Convolutional neural network tailored for the Fashion MNIST dataset
- Data augmentation with random cropping and horizontal flips
- Gradient clipping for training stability

### Reliability Index Strategy

The baselines were specifically used to validate the research gap addressed by the novel reliability index strategy:

- **Concept**: Instead of randomly selecting clients or using a fixed fraction, the reliability index intelligently selects clients based on their training performance metrics, prioritizing those with higher reliability
- **Enhanced Diversity**: The strategy is extended with a diversity-aware mechanism that balances model performance with client diversity
- **Implementation**: Extends Flower's FedAvg strategy with custom client selection logic
- **Validation**: Both CIFAR-10 and Fashion MNIST baselines were used to validate the effectiveness of this approach across different domains before applying it to network intrusion detection

#### Performance Comparison with FedAvg

The reliability index strategy was compared with the standard FedAvg algorithm to assess its performance advantages. Controlled experiments were conducted across both CIFAR-10 and Fashion MNIST datasets using identical client distributions, and hyperparameters, with the only difference being the client selection mechanism:

- **Robustness**: Demonstrated stronger resilience against non-IID data distributions and client heterogeneity

The experimental results demonstrated that reliability index-based client selection leads to:
1. Faster convergence rates
2. Higher final model accuracy
3. More efficient use of communication resources
4. Better performance in heterogeneous client environments

This strategy addresses the research gap in federated learning where traditional approaches don't account for client contribution quality in the selection process in data heterogeneous environments.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.5.1
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/almaznuman/ad-pytorch.git
   cd ad-pytorch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Download the UNSW-NB15 dataset files and place them in the `dataset` directory.

### Running the Platform

Launch the Streamlit UI:
```bash
python -m streamlit run frontend.py
```

The interface allows you to:
- Select model architecture (CNN-BiGRU Binary or Multiclass)
- Configure training parameters (rounds, local epochs, etc.)
- Choose federated learning strategy
- Enable/disable Weights & Biases tracking
- Start and monitor federated learning experiments

## 🔧 Configuration

Key configuration parameters in `pyproject.toml`:

- `num-server-rounds`: Number of federated learning rounds
- `local-epochs`: Number of training epochs per client
- `strategy-type`: Federated learning strategy (diversity, fedavg)
- `diversity-weight`: Weight for diversity-aware aggregation
- `use-wandb`: Enable Weights & Biases integration

## 📈 Results and Visualizations

Training results are stored in:
- `outputs/`: Training logs and model checkpoints
- `Visualizations/`: Performance visualizations and metrics
- `wandb/`: Weights & Biases experiment records (if enabled)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UNSW-NB15 dataset creators
- [Flower](https://flower.ai/) team for their federated learning framework and strategy implementations
- PyTorch community
 
