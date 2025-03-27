from datasets import load_dataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions

data_files = "./dataset/UNSW_NB15_training-set.csv"
dataset = load_dataset("csv", data_files=data_files)

# Initialize the partitioner with your dataset
# partitioner = IidPartitioner(num_partitions=num_partitions)
partitioner = DirichletPartitioner(
    num_partitions=10,
    partition_by="label",  # Your target column
    alpha=0.3,
    # alpha=5,  # Lower alpha = more heterogeneity
    self_balancing=False,
    seed=42
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