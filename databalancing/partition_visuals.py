from datasets import load_dataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.visualization import plot_label_distributions

data_files = "../dataset/UNSW_NB15_training-set.csv"
dataset = load_dataset("csv", data_files=data_files)

# Initialize the partitioner with your dataset
# partitioner = IidPartitioner(num_partitions=10)
partitioner = DirichletPartitioner(
    num_partitions=10,
    partition_by="label",  # Your target column
    alpha=0.4,
    # alpha=5,  # Lower alpha = more heterogeneity
    self_balancing=False,
    seed=42,
    # min_partition_size=1500
)

partitioner.dataset = dataset["train"]

figure, axis, dataframe = plot_label_distributions(
    partitioner=partitioner,
    label_name="label",
    legend=True,
)

# Save the plot
# figure.savefig('label_distribution.png')

print("\nLabel Distribution across partitions:")
print(dataframe)


# from datasets import load_dataset
# from flwr_datasets.partitioner import DirichletPartitioner
# from flwr_datasets.visualization import plot_label_distributions
#
# # Load Fashion-MNIST dataset
# dataset = load_dataset("zalando-datasets/fashion_mnist")
#
# alpha=0.5
#
# # Initialize the partitioner with same parameters as in task.py
# partitioner = DirichletPartitioner(
#     num_partitions=10,
#     partition_by="label",
#     alpha=alpha,  # Low alpha for high heterogeneity
#     seed=42
# )
#
# # Set the dataset for partitioning
# partitioner.dataset = dataset["train"]
#
# # Create the visualization
# figure, axis, dataframe = plot_label_distributions(
#     partitioner=partitioner,
#     label_name="label",
#     legend=True,
# )
#
# # Save the plot
# figure.savefig(f"fmnist_{alpha}_label_distribution.png")
#
# print("\nLabel Distribution across partitions:")
# print(dataframe)