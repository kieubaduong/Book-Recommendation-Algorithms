import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
import csv
import os

df_books = pd.read_csv("../../dataset/processed_dataset/books.csv")
df_book_features = pd.read_csv("../../dataset/book_features.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

book_features = np.zeros((len(df_books), 768))

for i, row in df_books.iterrows():
    isbn = row['isbn']
    book_feature = np.fromstring(df_book_features[df_book_features['isbn'] == isbn]['feature'].values[0][1:-1], sep=' ')
    book_features[i] = book_feature


knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
knn_model.fit(book_features)

result_file_path = "../../dataset/result.csv"

csv_exists = os.path.isfile(result_file_path)
existing_isbns = []

if csv_exists:
    with open(result_file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader) 
        for row in reader:
            existing_isbns.append(row[0]) 

with open(result_file_path, 'a', newline='') as file:
    writer = csv.writer(file, delimiter=';')

    if not csv_exists:
        writer.writerow(['isbn', 'top-10'])

    for isbn in df_books['isbn']:
        if csv_exists and isbn in existing_isbns:
            continue

        sample_features = df_book_features[df_book_features['isbn'] == isbn]['feature'].values[0]
        sample_features = np.fromstring(sample_features[1:-1], sep=' ')
        distances, indices = knn_model.kneighbors(sample_features.reshape(1, -1))
        top_10_isbns = ';'.join(df_books.iloc[indices[0][1:11]]['isbn'])
        writer.writerow([isbn, top_10_isbns])

print("Đã lưu kết quả tìm kiếm vào file result.csv")