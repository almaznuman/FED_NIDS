import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

import torch

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


def preprocessData(train_path, test_path):
    print("Loading and preprocessing data...")

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    data = pd.concat([train_df, test_df], ignore_index=True)

    # Define target variables
    multiclass = 'attack_cat'
    binary = 'label'

    # Label encode categorical features
    label_encoder = LabelEncoder()
    if multiclass in data.columns:
        data[multiclass] = label_encoder.fit_transform(data[multiclass])

    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Apply normalization to numerical features
    scaler = MinMaxScaler()
    features_to_scale = data.columns.difference([multiclass, binary])
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Extract features and targets
    X = data.drop(columns=[binary, multiclass])
    y_binary = data[binary]
    y_multi_class = data[multiclass]

    return X, y_binary, y_multi_class


def selectFeatures(x, y_binary, y_multi_class):
    """Perform feature selection using RandomForest and RFECV"""
    print("Performing feature selection...")

    # Split data for feature selection
    X_train, X_test, y_train_binary, y_test_binary, y_train_multi_class, y_test_multi_class = train_test_split(
        x, y_binary, y_multi_class, test_size=0.2, random_state=42)

    # Step 1: Initial Feature Selection with RandomForest for Binary Classification
    random_forrest_binary = RandomForestClassifier(random_state=42)
    random_forrest_binary.fit(X_train, y_train_binary)

    # Get feature importances
    feature_importance_binary = random_forrest_binary.feature_importances_
    binary_columns = np.argsort(feature_importance_binary)[::-1]

    # Select top features (e.g., top 50%)
    binary_criteria = np.median(feature_importance_binary)
    b_selected_features = binary_columns[feature_importance_binary > binary_criteria]

    # Print selected features for binary classification
    selected_features_binary = X_train.columns[b_selected_features]
    print(f"Selected {len(selected_features_binary)} features for binary classification: ")
    print(selected_features_binary.tolist())

    # Step 2: Initial Feature Selection with RandomForest for Multi-Class Classification
    random_forrest_multi_class = RandomForestClassifier(random_state=42)
    random_forrest_multi_class.fit(X_train, y_train_multi_class)

    # Get feature importances
    feature_importances_multi_class = random_forrest_multi_class.feature_importances_
    multi_columns = np.argsort(feature_importances_multi_class)[::-1]

    # Select top features (e.g., top 50%)
    multi_criteria = np.median(feature_importances_multi_class)
    m_selected_features = multi_columns[feature_importances_multi_class > multi_criteria]

    # Print selected features for multi-class classification
    selected_features_multi_class = X_train.columns[m_selected_features]
    print(f"Selected {len(selected_features_multi_class)} features for multi-class classification: ")
    print(selected_features_multi_class.tolist())

    # Step 3: RFECV with Logistic Regression for Binary Classification
    log_reg_binary = LogisticRegression(max_iter=1000, random_state=42)
    rfecv_binary = RFECV(estimator=log_reg_binary, step=1, cv=5, scoring='accuracy')

    # Apply RFECV to the subset of features selected by RandomForest
    X_train_rf_binary = X_train.iloc[:, b_selected_features]
    rfecv_binary.fit(X_train_rf_binary, y_train_binary)

    # Get the selected feature indices after RFECV
    selected_indices_rfecv_binary = np.array(b_selected_features)[rfecv_binary.get_support()]
    selected_features_rfecv_binary = x.columns[selected_indices_rfecv_binary]

    print(f"Final {len(selected_features_rfecv_binary)} features after RFECV (Binary): ")
    print(selected_features_rfecv_binary.tolist())

    # Step 4: RFECV with Logistic Regression for Multi-Class Classification
    log_reg_multi_class = LogisticRegression(max_iter=1000, random_state=42)
    rfecv_multi_class = RFECV(estimator=log_reg_multi_class, step=1, cv=5, scoring='accuracy')

    # Apply RFECV to the subset of features selected by RandomForest
    X_train_rf_multi_class = X_train.iloc[:, m_selected_features]
    rfecv_multi_class.fit(X_train_rf_multi_class, y_train_multi_class)

    # Get the selected feature indices after RFECV
    selected_indices_rfecv_multi_class = np.array(m_selected_features)[rfecv_multi_class.get_support()]
    selected_features_rfecv_multi_class = x.columns[selected_indices_rfecv_multi_class]

    print(f"Final {len(selected_features_rfecv_multi_class)} features after RFECV (Multi-Class): ")
    print(selected_features_rfecv_multi_class.tolist())

    common_columns = sorted(list(set(selected_indices_rfecv_binary) | set(selected_indices_rfecv_multi_class)))
    print(f"Using {len(common_columns)} common features for both tasks")

    print(f"Selected features: {common_columns}")


def main():

    train_path = 'dataset/UNSW_NB15_training-set.csv'
    test_path = 'dataset/UNSW_NB15_testing-set.csv'

    # Load and preprocess data
    X, y_binary, y_multi_class = preprocessData(train_path, test_path)

    # Perform feature selection
    selectFeatures(X, y_binary, y_multi_class)

if __name__ == "__main__":
    main()
