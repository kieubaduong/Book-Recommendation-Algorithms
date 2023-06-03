import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

df_books = pd.read_csv("../dataset/processed_books.csv", delimiter=",", usecols = [0,1,2,3,4], dtype={'Year-Of-Publication':object})

df_books_subset = df_books.head(10)

weights = {
    'title': 0.5,
    'author': 0.3,
    'publish_year': 0.2,
    'publisher': 0.1
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(row):
    title = row['Book-Title']
    author = row['Book-Author']
    publish_year = row['Year-Of-Publication']
    publisher = row['Publisher']

    title_inputs = tokenizer(title, padding=True, truncation=True, max_length=128, return_tensors='pt')
    title_outputs = model(**title_inputs)
    title_features = title_outputs['pooler_output'].detach().numpy().squeeze()

    author_inputs = tokenizer(author, padding=True, truncation=True, max_length=128, return_tensors='pt')
    author_outputs = model(**author_inputs)
    author_features = author_outputs['pooler_output'].detach().numpy().squeeze()

    year_inputs = tokenizer(publish_year, padding=True, truncation=True, max_length=128, return_tensors='pt')
    year_outputs = model(**year_inputs)
    year_features = year_outputs['pooler_output'].detach().numpy().squeeze()

    publisher_inputs = tokenizer(publisher, padding=True, truncation=True, max_length=128, return_tensors='pt')
    publisher_outputs = model(**publisher_inputs)
    publisher_features = publisher_outputs['pooler_output'].detach().numpy().squeeze()
    
    title_features *= weights['title']
    author_features *= weights['author']
    year_features *= weights['publish_year']
    publisher_features *= weights['publisher']

    combined_features = np.concatenate((title_features, author_features, year_features, publisher_features))
    return combined_features


book_features = np.zeros((len(df_books_subset), 768*4)) 

for i, row in df_books_subset.iterrows():
    book_features[i] = extract_features(row)
    
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(book_features)

knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(book_features)

sample_book = {'Book-Title': 'Slumdog Millionaire', 'Book-Author': 'Vikas Swarup', 'Year-Of-Publication': '2008', 'Publisher': 'HarperCollins'}
sample_features = extract_features(sample_book)

distances, indices = knn_model.kneighbors(sample_features.reshape(1, -1))

print("distance" + str(distances))

for index in indices:
    book_info = df_books_subset.iloc[index]
    print(book_info)