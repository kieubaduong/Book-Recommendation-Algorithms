import pandas as pd

# Đọc dataset user vào dataframe
df_users = pd.read_csv('../../dataset/processed_dataset/users.csv')

# Đọc dataset rating vào dataframe
df_ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv')


# Ghép hai dataframe dựa trên cột 'user-id'
df_merged = pd.merge(df_ratings, df_users, on='user-id', how='inner')

# Lấy danh sách các user-id có trong rating dataset
user_ids_in_ratings = df_ratings['user-id'].unique()

# Lọc các user trong user dataset dựa trên user-id có trong rating dataset
df_filtered_users = df_users[df_users['user-id'].isin(user_ids_in_ratings)]

df_filtered_users.to_csv('../../dataset/processed_dataset/users.csv', index=False)
