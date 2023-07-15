import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import csv
import os

df_books = pd.read_csv("../../dataset/processed_dataset/books.csv")

weights = {
    'title': 1.0,
    'author': 0.8,
    'year': 0.5,
    'publisher': 0.7,
    'tags' : 0.8,
    'description' : 0.9,
    'genres' : 0.6,
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(isbn):
    row = df_books[df_books['isbn'] == isbn].iloc[0]
    title = row['title']
    author = row['author']
    year = str(row['year'])
    publisher = row['publisher']
    tags = row['tags']
    description = row['description']
    genres = row['genres']

    title_inputs = tokenizer(title, padding=True, truncation=True, max_length=128, return_tensors='pt')
    title_outputs = model(**title_inputs)
    title_features = title_outputs['pooler_output'].detach().numpy().squeeze()

    author_inputs = tokenizer(author, padding=True, truncation=True, max_length=128, return_tensors='pt')
    author_outputs = model(**author_inputs)
    author_features = author_outputs['pooler_output'].detach().numpy().squeeze()

    year_inputs = tokenizer(year, padding=True, truncation=True, max_length=128, return_tensors='pt')
    year_outputs = model(**year_inputs)
    year_features = year_outputs['pooler_output'].detach().numpy().squeeze()

    publisher_inputs = tokenizer(publisher, padding=True, truncation=True, max_length=128, return_tensors='pt')
    publisher_outputs = model(**publisher_inputs)
    publisher_features = publisher_outputs['pooler_output'].detach().numpy().squeeze()

    tags_inputs = tokenizer(tags, padding=True, truncation=True, max_length=128, return_tensors='pt')
    tags_outputs = model(**tags_inputs)
    tags_features = tags_outputs['pooler_output'].detach().numpy().squeeze()

    description_inputs = tokenizer(description, padding=True, truncation=True, max_length=128, return_tensors='pt')
    description_outputs = model(**description_inputs)
    description_features = description_outputs['pooler_output'].detach().numpy().squeeze()

    genres_inputs = tokenizer(genres, padding=True, truncation=True, max_length=128, return_tensors='pt')
    genres_outputs = model(**genres_inputs)
    genres_features = genres_outputs['pooler_output'].detach().numpy().squeeze()

    title_features *= weights['title']
    author_features *= weights['author']
    year_features *= weights['year']
    publisher_features *= weights['publisher']
    tags_features *= weights['tags']
    description_features *= weights['description']
    genres_features *= weights['genres']

    combined_features = np.concatenate((title_features, author_features, year_features, publisher_features, tags_features, description_features, genres_features))
    return combined_features

print(df_books[df_books['isbn'] == '0192126040'].iloc[0])
print(extract_features('0192126040'))
print(extract_features('0192126040').shape)

# output_path = "/content/drive/MyDrive/Dataset/featured/book_features_v2.csv"

# existing_isbns = set()

# if os.path.exists(output_path):
#     with open(output_path, 'r') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         next(reader)  # Skip header row
#         for row in reader:
#             existing_isbns.add(row[0])

# with open(output_path, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     if csvfile.tell() == 0:
#         writer.writerow(['isbn', 'feature'])
#     for isbn in df_books['isbn']:
#         if isbn in existing_isbns:
#             continue
#         features = extract_features(isbn)
#         writer.writerow([isbn, features])