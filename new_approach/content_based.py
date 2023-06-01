import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

# df_books = pd.read_csv("../dataset/processed_books.csv", delimiter=",", usecols = [0,1,2,3,4], dtype={'Year-Of-Publication':object})

book_features = pd.read_csv("../test.csv")

import json


# dictionary_data = json.loads(book_features['features'][0])

# print(dictionary_data)


knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')

# Fit dữ liệu vào mô hình KNN
knn_model.fit(book_features['features'])

# joblib.dump(knn_model, 'knn_model.pkl')

# Lấy quyển sách mới
# new_book = # ... (lấy feature của quyển sách mới)

# # Tìm k neighbors gần nhất của quyển sách mới
# distances, indices = knn_model.kneighbors([new_book])

# # Lấy thông tin về 10 quyển sách giống nhất
# similar_books = df.iloc[indices[0]]