import pandas as pd

df = pd.read_csv('../../dataset/processed_dataset/users.csv')

df_unique = df.drop_duplicates(subset='user-id', keep='first')

df_unique.to_csv("../../dataset/processed_dataset/users.csv", index=False)
