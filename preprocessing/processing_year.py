import pandas as pd
import numpy as np

df = pd.read_csv('../dataset/processed_dataset/books.csv')

median_year = np.median(df['year'])
df['year'] = df['year'].replace(0, median_year)
df['year'] = df['year'].astype(int)

df.to_csv('../dataset/processed_dataset/books.csv', index=False)