import pandas as pd

df = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

df = df.dropna()

df.to_csv('../../dataset/processed_dataset/ratings.csv', index=False)
