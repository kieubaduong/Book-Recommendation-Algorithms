import numpy as np
import pandas as pd
# !pip install transformers
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors
import ast
import json


df_rating = pd.read_csv("/content/drive/MyDrive/Dataset/processed/ratings.csv")
df_book = pd.read_csv("/content/drive/MyDrive/Dataset/processed/books.csv")
df_user = pd.read_csv("/content/drive/MyDrive/Dataset/processed/users.csv")
book_features_v2 = pd.read_csv("/content/drive/MyDrive/Dataset/featured/book_features_v2.csv")
user_features_v2 = pd.read_csv("/content/drive/MyDrive/Dataset/featured/user_features_v2.csv")

def convert_string_to_vector(string):
    string = string.strip().lstrip("[").rstrip("]")
    values = [float(value) for value in string.split()]
    return np.array(values)

user_features = np.vstack(user_features_v2['feature'].apply(convert_string_to_vector))

neighbors = 10
knn = NearestNeighbors(n_neighbors=neighbors, metric='cosine')
knn.fit(user_features)

user_id = 242
target_user_features = convert_string_to_vector(user_features_v2[user_features_v2['user-id'] == user_id]['feature'].values[0])

distances, indices = knn.kneighbors(target_user_features.reshape(1, -1), n_neighbors=neighbors+1)

indices = indices.flatten()[1:]
nearest_users = df_user.loc[indices]

recommended_books = []

for _, user_row in nearest_users.iterrows():
    user_id = user_row['user-id']
    user_ratings = df_rating[(df_rating['user-id'] == user_id) & (df_rating['book-rating'] >= 3)]
    liked_books = user_ratings['isbn'].tolist()
    remaining_space = 10 - len(recommended_books)
    if remaining_space > 0:
        if len(liked_books) > remaining_space:
            recommended_books.extend(liked_books[:remaining_space])
        else:
            recommended_books.extend(liked_books)
    if len(recommended_books) >= 10:
        break

for book_id in recommended_books:
    book_info = df_book[df_book['isbn'] == book_id]
    if not book_info.empty:
        title = book_info['title'].values[0]
        author = book_info['author'].values[0]
        isbn = book_info['isbn'].values[0]
        print(f"Book: {title} - Author: {author} - ISBN: {isbn}")