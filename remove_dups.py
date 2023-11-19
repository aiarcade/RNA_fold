import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('../test_sequences.csv')

print(len(df))
# Remove the first 176 rows
