import pandas as pd


df_rating = pd.read_csv('../../dataset/processed_dataset/ratings.csv')
df_user = pd.read_csv('../../dataset/processed_dataset/users.csv')

merged_df = pd.merge(df_rating, df_user, on='user-id', how='inner')
cleaned_df = merged_df[['user-id', 'isbn', 'book-rating']]

cleaned_df.to_csv('../../dataset/processed_dataset/ratings.csv', index=False)
