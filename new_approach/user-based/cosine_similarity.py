import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Đọc dữ liệu từ file user_features_v2.csv
df_user_features = pd.read_csv("/content/drive/MyDrive/Dataset/featured/user_features_v2.csv")
df_rating = pd.read_csv("/content/drive/MyDrive/Dataset/processed/ratings.csv")
df_book = pd.read_csv("/content/drive/MyDrive/Dataset/processed/books.csv")

df_user_features = df_user_features.drop_duplicates(subset='feature', keep='first')
user_features = np.array([np.fromstring(features[1:-1], dtype=float, sep=' ') for features in df_user_features['feature']])
user_ids = df_user_features['user-id'].values

# Chuẩn hóa các vector đặc trưng của users
normalized_user_features = normalize(user_features)

# Tạo vector đặc trưng cho user input và chuẩn hóa nó
input_vector = np.array([1.0, 2.0, 3.0])  # Vector đặc trưng của user input
normalized_input_vector = normalize(input_vector.reshape(1, -1))

# Tính cosine similarity giữa user input và các users
cosine_similarities = cosine_similarity(normalized_input_vector, normalized_user_features)

# Sắp xếp các users theo độ tương đồng giảm dần
sorted_indices = np.argsort(cosine_similarities)[0][::-1]
nearest_users = user_ids[sorted_indices]

recommended_books = []
book_seen = set()

for user_id in nearest_users:
    user_ratings = df_rating.loc[(df_rating['user-id'] == user_id) & (~df_rating['isbn'].isin(book_seen)), 'isbn']
    liked_books = user_ratings.tolist()
    remaining_space = 10 - len(recommended_books)
    
    for book_id in liked_books:
        if book_id not in book_seen and len(recommended_books) < 10:
            recommended_books.append(book_id)
            book_seen.add(book_id)
    
    if len(recommended_books) >= 10:
        break

for book_id in recommended_books:
    book_info = df_book[df_book['isbn'] == book_id]
    if not book_info.empty:
        title = book_info['title'].values[0]
        author = book_info['author'].values[0]
        isbn = book_info['isbn'].values[0]
        print(f"Book: {title} - Author: {author} - ISBN: {isbn}")
