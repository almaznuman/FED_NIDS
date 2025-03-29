#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script to load and use the optimized lightweight CNN+BiGRU model
"""

import os
import numpy as np
import torch
import json
from cnnbigru_binary_optuna import CNNBiGRUBinaryModel, load_data

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_optimized_model(model_path='binary_model_hyperopt/best_model.pth'):
    """Load the optimized model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run hyperparameter tuning first.")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    hyperparams = checkpoint['hyperparameters']
    input_dim = checkpoint['input_dim']
    
    # Create model with the optimal hyperparameters
    model = CNNBiGRUBinaryModel(
        input_dim=input_dim,
        cnn_filters=hyperparams['cnn_filters'],
        kernel_size=hyperparams['kernel_size'],
        hidden_dim=hyperparams['hidden_dim'],
        fc_units=hyperparams['fc_units'],
        dropout=hyperparams['dropout'],
        bidirectional=hyperparams['bidirectional'],
        num_layers=hyperparams['num_layers']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Print model details
    print("\nOptimized Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model, hyperparams

def load_original_model(input_dim, hidden_dim=50):
    """Load the original model configuration for comparison."""
    model = CNNBiGRUBinaryModel(
        input_dim=input_dim,
        cnn_filters=64,
        kernel_size=3,
        hidden_dim=hidden_dim,
        fc_units=100,
        dropout=0.01,
        bidirectional=True,
        num_layers=1
    ).to(device)
    
    print("\nOriginal Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model

def compare_models(optimized_params, output_dir='binary_model_hyperopt'):
    """Print a comparison between the original and optimized model."""
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
        
        print("\nModel Comparison:")
        print(f"Original Model Size:  {comparison['original_model_size']} parameters")
        print(f"Optimized Model Size: {comparison['optimized_model_size']} parameters")
        print(f"Size Reduction:       {comparison['size_reduction_percent']:.2f}%")
        print(f"Accuracy:             {comparison['accuracy']:.4f}")
        print(f"F1 Score:             {comparison['f1_score']:.4f}")
    else:
        print("\nComparison data not found. Run the complete hyperparameter tuning first.")
    
    print("\nOptimized Model Hyperparameters:")
    for param, value in optimized_params.items():
        print(f"  {param}: {value}")

def predict_sample(model, input_tensor):
    """Make prediction for a single sample."""
    model.eval()
    with torch.no_grad():
        # Forward pass
        output = model(input_tensor)
        
        # Convert output to prediction
        prediction = (output > 0.5).int().cpu().numpy()
        confidence = output.cpu().numpy()
        
    return prediction, confidence

def main():
    # Set paths to your data files
    train_path = '../../../dataset/UNSW_NB15_testing-set.csv'
    test_path = '../../../dataset/UNSW_NB15_testing-set.csv'
    
    try:
        # Load optimized model
        optimized_model, hyperparams = load_optimized_model()
        
        # Load a few samples from the dataset for demonstration
        X_train, X_test, y_train, y_test = load_data(train_path, test_path)
        input_dim = X_train.shape[1]
        
        # Load original model for comparison
        original_model = load_original_model(input_dim)
        
        # Compare the models
        compare_models(hyperparams)
        
        # Prepare a small batch of test data
        num_samples = 5
        test_samples = torch.FloatTensor(X_test[:num_samples]).to(device)
        true_labels = y_test[:num_samples].values
        
        # Make predictions with both models
        print("\nPredictions on sample data:")
        print(f"{'Sample':^8} | {'True Label':^10} | {'Original Pred':^12} | {'Optimized Pred':^14} | {'Orig Conf':^10} | {'Opt Conf':^10}")
        print("-" * 80)
        
        for i in range(num_samples):
            sample = test_samples[i:i+1]
            
            # Predict with original model
            orig_pred, orig_conf = predict_sample(original_model, sample)
            
            # Predict with optimized model
            opt_pred, opt_conf = predict_sample(optimized_model, sample)
            
            # Print results
            print(f"{i:^8} | {true_labels[i]:^10} | {orig_pred[0][0]:^12} | {opt_pred[0][0]:^14} | {orig_conf[0][0]:.4f} | {opt_conf[0][0]:.4f}")
        
        print("\nModel Inference Time Comparison (10 runs on the same batch):")
        import time
        
        # Create a batch for timing
        batch_size = 32
        timing_batch = torch.FloatTensor(X_test[:batch_size]).to(device)
        
        # Time original model
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = original_model(timing_batch)
        original_time = (time.time() - start_time) / 10
        
        # Time optimized model
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = optimized_model(timing_batch)
        optimized_time = (time.time() - start_time) / 10
        
        # Calculate speedup
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"Original model:  {original_time*1000:.2f} ms per batch")
        print(f"Optimized model: {optimized_time*1000:.2f} ms per batch")
        print(f"Speedup:         {speedup:.2f}x")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the hyperparameter tuning script first.")

if __name__ == "__main__":
    main() 