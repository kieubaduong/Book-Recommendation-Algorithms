import pandas as pd

df = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

df = df.drop_duplicates(subset=['user-id', 'isbn'])

df.to_csv('../../dataset/processed_dataset/ratings.csv', index=False)