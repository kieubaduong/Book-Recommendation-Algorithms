import pandas as pd

df_rating = pd.read_csv('../../dataset/processed_dataset/ratings.csv')
df_user = pd.read_csv('../../dataset/processed_dataset/users.csv')

user_ids_in_rating = df_rating['user-id'].unique()  

df_user_filtered = df_user[df_user['user-id'].isin(user_ids_in_rating)]

df_user_filtered.to_csv('../../dataset/processed_dataset/users.csv', index=False)

