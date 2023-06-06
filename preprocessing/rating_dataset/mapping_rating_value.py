import pandas as pd

df_ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

# Tạo bảng ánh xạ giá trị book-rating từ cũ sang mới
rating_mapping = {
    9: 5,
    10: 5,
    8: 4,
    7: 4,
    6: 3,
    5: 3,
    4: 2,
    3: 2,
    2: 1,
    1: 1,
    0: 1
}

# Ánh xạ giá trị book-rating theo bảng ánh xạ
df_ratings['book-rating'] = df_ratings['book-rating'].map(rating_mapping)

df_ratings.to_csv('../../dataset/processed_dataset/ratings.csv', index=False)
