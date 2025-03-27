from sklearn.model_selection import train_test_split
import pandas as pd
# Read Unsw-nb15 datasets combine them and shuffle them
# Read the datasets
df1 = pd.read_csv('../dataset/UNSW_NB15_training-set.csv')
df2 = pd.read_csv('../dataset/UNSW_NB15_testing-set.csv')

# Combine the datasets
df = pd.concat([df1, df2])

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# separate the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# print label distribution
print(train_data['label'].value_counts())
print(test_data['label'].value_counts())

# save the training and testing sets to separate csv files
train_data.to_csv('../dataset/train_data.csv', index=False)
test_data.to_csv('../dataset/test_data.csv', index=False)

# summarize the dataset
print(train_data.describe())
print(test_data.describe())
print("###############################")
# print the number of unique values for each column
print(train_data.nunique())
print(test_data.nunique())

