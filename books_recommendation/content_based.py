import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import missingno as msno

df_books = pd.read_csv("../dataset/processed_books.csv", delimiter=",", usecols = [0,1,2,3,4], dtype={'Year-Of-Publication':object})
# df_users = pd.read_csv("../dataset/processed_users.csv")
# df_rating = pd.read_csv("../dataset/processed_rating.csv")

df_books_subset = df_books.head(400)

from transformers import BertTokenizer, BertModel

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
    
    input_text = f"{title} {author} {publish_year} {publisher}"
    inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length', max_length=128, return_tensors='pt')
    
    outputs = model(**inputs)
    hidden_state = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    
    features = {
        'title': hidden_state * weights['title'],
        'author': hidden_state * weights['author'],
        'publish_year': hidden_state * weights['publish_year'],
        'publisher': hidden_state * weights['publisher']
    }
    return features

df_books_subset['features'] = df_books_subset.apply(extract_features, axis=1)

import os
folder_path = os.path.dirname(os.getcwd())
file_path = os.path.join(folder_path, 'features.csv')

df_books_subset.to_csv(file_path, index=False)


