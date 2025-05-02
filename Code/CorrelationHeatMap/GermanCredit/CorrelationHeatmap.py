# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the GermanCredit dataset from a local file
# Update the path to your local dataset
file_path = r'C:\\Users\\Lenovo\\Downloads\\german_credit_data (2).csv'
credit = pd.read_csv(file_path)

# Check the first few rows of the dataset
print("Dataset Preview:")
print(credit.head())

# Preprocessing the Data
# Convert non-numeric columns to numeric if necessary
# For example, we encode categorical variables to numeric values
from sklearn.preprocessing import LabelEncoder

categorical_cols = credit.select_dtypes(include=['object']).columns
for col in categorical_cols:
    credit[col] = LabelEncoder().fit_transform(credit[col])

# Dropping non-relevant columns (if any, like 'Index')
if 'Index' in credit.columns:
    credit = credit.drop(columns=['Index'])

# Compute the correlation matrix
correlation_matrix = credit.corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Generate a Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of GermanCredit Dataset", fontsize=16)
plt.tight_layout()

# Save the heatmap as a file (optional)
plt.savefig('correlation_heatmap.png')

# Display the heatmap
plt.show()