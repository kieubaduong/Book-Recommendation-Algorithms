import pandas as pd
import re

df = pd.read_csv('../../dataset/processed_dataset/users.csv')

df_filtered = df[df['age'] >= 3]

df_filtered.to_csv("../../dataset/processed_dataset/users.csv", index=False)