# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the GermanCredit dataset from the provided URL
# url = 'https://raw.githubusercontent.com/S-B-Iqbal/Test/master/german_credit_data.csv'
credit = pd.read_csv('C:\\Users\\Lenovo\\Downloads\\german_credit_data (1).csv', header=0, names=['Index', 'Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                                           'Checking account', 'Credit amount', 'Duration', 'Purpose', 'default'])

# Preprocessing: Fill missing values
credit['Saving accounts'] = credit['Saving accounts'].fillna(value='NA')
credit['Checking account'] = credit['Checking account'].fillna(value='NA')

# Function to create and save countplots for each categorical variable
def generate_countplot(dataframe, column_name, target='default'):
    """
    This function generates and displays a countplot for a given column.

    Parameters:
        dataframe: DataFrame containing the dataset
        column_name: The categorical column to analyze
        target: The target column for the hue parameter (default is 'default')
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column_name, data=dataframe, hue=target, palette='viridis')
    plt.title(f'Countplot for {column_name} vs {target}', fontsize=14)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title=target, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{column_name}_countplot.png')  # Save plot as a file
    plt.show()

# List of categorical columns for visualization
categorical_columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']

# Generate countplots for each column
for column in categorical_columns:
    generate_countplot(credit, column)