import pandas as pd


df_rating = pd.read_csv('../dataset/processed_dataset/ratings.csv')
df_user = pd.read_csv('../dataset/processed_dataset/users.csv')

rating_users = set(df_rating['user-id'].unique())
user_users = set(df_user['user-id'].unique())

missing_users = rating_users - user_users

if len(missing_users) > 0:
    print("Có người dùng trong tập dữ liệu rating không tồn tại trong tập dữ liệu user:")
    for user_id in missing_users:
        print(user_id)
else:
    print("Không có người dùng trong tập dữ liệu rating không tồn tại trong tập dữ liệu user.")
