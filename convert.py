import pandas as pd
import numpy as np

df = pd.read_csv('train_data.csv')
for col in df.columns:
    if df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float32)
df.to_parquet('train_data.parquet')
