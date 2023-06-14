import numpy as np
import pandas as pd
# !pip install transformers
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors
import ast
import json
import csv
import os

df_rating = pd.read_csv("/content/drive/MyDrive/Dataset/processed/ratings.csv")
df_book = pd.read_csv("/content/drive/MyDrive/Dataset/processed/books.csv")
df_user = pd.read_csv("/content/drive/MyDrive/Dataset/processed/users.csv")
book_features_v2 = pd.read_csv("/content/drive/MyDrive/Dataset/featured/book_features_v2.csv")
    
# df_user = df_user.head(11)

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

existing_users = set()
output_path = "/content/drive/MyDrive/Dataset/featured/user_features_v2.csv"
column_names = ['user-id', 'feature']

# Kiểm tra xem file tồn tại hay không
if os.path.isfile(output_path):
    with open(output_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Bỏ qua dòng tiêu đề
        for row in reader:
            user_id = row[0]
            existing_users.add(int(user_id))
else:
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

with open(output_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for _, user_row in df_user.iterrows():
        user_id = user_row['user-id']
        if user_id in existing_users:
            continue
        user_features = extract_user_features(user_id)
        writer.writerow([user_id] + user_features.tolist())

print(f"User features extracted and saved to {output_path}.")