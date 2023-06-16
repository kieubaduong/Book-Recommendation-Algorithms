import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Đọc dữ liệu từ file user_features_v2.csv
df_user_features = pd.read_csv("/content/drive/MyDrive/Dataset/featured/user_features_v2.csv")

df_user_features = df_user_features.drop_duplicates(subset='feature', keep='first')
user_features = np.array([np.fromstring(features[1:-1], dtype=float, sep=' ') for features in df_user_features['feature']])
user_ids = df_user_features['user-id'].values

# Chuẩn hóa các vector đặc trưng của users
normalized_user_features = normalize(user_features)

# Tạo vector đặc trưng cho user input và chuẩn hóa nó
input_vector = np.array([1.0, 2.0, 3.0])  # Vector đặc trưng của user input
normalized_input_vector = normalize(input_vector.reshape(1, -1))

# Tìm k nearest neighbors sử dụng cosine similarity
k = 10
nn = NearestNeighbors(n_neighbors=k, metric='cosine')
nn.fit(normalized_user_features)
distances, indices = nn.kneighbors(normalized_input_vector)

# In ra các user gần nhất với user input
nearest_users = user_ids[indices[0]]
for user_id in nearest_users:
    print(user_id)