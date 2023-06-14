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
    
df_user = df_user.head(11)

def convert_to_vector(x):
    x = x.strip().rstrip("\\n")
    return np.fromstring(x[1:-1], sep=' ')

book_features_v2['feature'] = book_features_v2['feature'].apply(convert_to_vector)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
default_features = np.zeros((768,))

def extract_features(book_info, rating):
    isbn = book_info['isbn'].values[0]
    features = book_features_v2[book_features_v2['isbn'] == isbn]['feature'].values[0]
    weighted_features = features * (rating - 3)
    return weighted_features

def extract_user_features(user_id):
    user_ratings = df_rating[df_rating['user-id'] == user_id]
    rated_books = user_ratings['isbn'].tolist()

    book_features = []
    
    if len(rated_books) == 0:
        return default_features
    else:
        for book_isbn in rated_books:
            rating_info = df_rating[(df_rating['user-id'] == user_id) & (df_rating['isbn'] == book_isbn)]
            if not rating_info.empty:
                rating = rating_info['book-rating'].values[0]
                book_info = df_book[df_book['isbn'] == book_isbn]
                if not book_info.empty:
                    features = extract_features(book_info, rating)
                    book_features.append(features)

        book_features_array = np.array(book_features)
        user_features = np.prod(book_features_array, axis=0)
        return user_features

user_features = []

user_features = df_user['user-id'].apply(extract_user_features)
user_features_array = np.array(user_features.tolist())

neighbors = 10
knn = NearestNeighbors(n_neighbors=neighbors, metric='cosine')
knn.fit(user_features_array)

user_id = 276847
target_user_features = extract_user_features(user_id)

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
        print(f"Book: {title} - Author: {author}")