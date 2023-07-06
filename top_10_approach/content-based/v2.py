import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

df_books = pd.read_csv("../../dataset/processed_dataset/books.csv")

df_books_subset = df_books.head(11)

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


book_features = np.zeros((len(df_books_subset), 768*7)) 

for i, row in df_books_subset.iterrows():
    isbn = row['isbn']
    book_features[i] = extract_features(isbn)

# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(book_features)

knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
knn_model.fit(book_features)

sample_features = extract_features('0192126040')

distances, indices = knn_model.kneighbors(sample_features.reshape(1, -1))

for index in indices[0][1:]:
    book_info = df_books_subset.iloc[index]
    print(book_info['isbn'])