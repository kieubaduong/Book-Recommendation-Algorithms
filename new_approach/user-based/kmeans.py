import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import ast

# Đọc dữ liệu từ file user_features_v2.csv
df_user_features = pd.read_csv("/content/drive/MyDrive/Dataset/featured/user_features_v2.csv")

df_user_features = df_user_features.drop_duplicates(subset='feature', keep='first')
user_features = np.array([np.fromstring(features[1:-1], dtype=float, sep=' ') for features in df_user_features['feature']])
user_ids = df_user_features['user-id'].values

# Khởi tạo K-means
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
cluster_labels = kmeans.fit_predict(user_features)

# Tạo vector đặc trưng cho input
input_vector = np.array([1.0, 2.0, 3.0])  # Vector đặc trưng của input

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
for user_id in nearest_users:
    print(user_id)
