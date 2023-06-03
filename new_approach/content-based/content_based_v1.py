import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel

df_books = pd.read_csv("../dataset/processed_books.csv", delimiter=",", usecols = [0,1,2,3,4], dtype={'Year-Of-Publication':object})

df_books_subset = df_books.head(100)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(row):
    title = row['Book-Title']
    author = row['Book-Author']
    publish_year = row['Year-Of-Publication']
    publisher = row['Publisher']

    input_text = f"{title} {author} {publish_year} {publisher}"
    inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length', max_length=128, return_tensors='pt')

    outputs = model(**inputs)

    hidden_state = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return hidden_state

book_features = np.zeros((len(df_books_subset), 768))

for i, row in df_books_subset.iterrows():
    book_features[i] = extract_features(row)

knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(book_features)

sample_book = {'Book-Title': 'Slumdog Millionaire', 'Book-Author': 'Vikas Swarup', 'Year-Of-Publication': '2008', 'Publisher': 'HarperCollins'}
sample_features = extract_features(sample_book)

distances, indices = knn_model.kneighbors(sample_features.reshape(1, -1))

print("distance" + str(distances))
print("indices" + str(indices))