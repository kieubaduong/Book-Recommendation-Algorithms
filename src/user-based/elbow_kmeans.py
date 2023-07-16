import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Đọc dữ liệu từ file user_features_v2.csv
df_user_features = pd.read_csv("user_features_v2.csv")
df_rating = pd.read_csv("ratings.csv")
df_book = pd.read_csv("books.csv")

df_user_features = df_user_features.drop_duplicates(subset='feature', keep='first')
user_features = np.array([np.fromstring(features[1:-1], dtype=float, sep=' ') for features in df_user_features['feature']])
user_ids = df_user_features['user-id'].values

# Tạo vector đặc trưng cho input
input_vector = np.array([1.0, 2.0, 3.0])  # Vector đặc trưng của input

# Khởi tạo K-means
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
cluster_labels = kmeans.fit_predict(user_features)
# Tìm cụm gần nhất cho input
nearest_cluster = kmeans.predict([input_vector])[0]

# Lấy danh sách các user trong cụm gần nhất
users_in_cluster = user_ids[cluster_labels == nearest_cluster]

# Tạo ma trận user features chỉ với các user trong cụm gần nhất
users_in_cluster_features = user_features[cluster_labels == nearest_cluster]

# Tìm k nearest neighbors trong cụm gần nhất
k = 10
nn = NearestNeighbors(n_neighbors=k)
nn.fit(users_in_cluster_features)
distances, indices = nn.kneighbors([input_vector])

# In ra các user gần nhất trong cụm gần nhất
nearest_users = users_in_cluster[indices[0]]
    
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
        print(f"ISBN: {isbn}")
