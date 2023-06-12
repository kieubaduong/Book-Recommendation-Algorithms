import pandas as pd

# Đọc tập dữ liệu đánh giá
df_ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

# Tính tổng số lượng đánh giá của mỗi người dùng
user_rating_counts = df_ratings['user-id'].value_counts()

# Lọc các đánh giá dựa trên điều kiện tổng số lượng đánh giá của người dùng lớn hơn hoặc bằng 2
filtered_ratings = df_ratings[df_ratings['user-id'].isin(user_rating_counts[user_rating_counts >= 2].index)]

# Lưu trữ dữ liệu đã lọc vào tập dữ liệu mới
filtered_ratings.to_csv('../../dataset/processed_dataset/ratings.csv', index=False)

# print(filtered_ratings.info())