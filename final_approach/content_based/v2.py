import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

df_books = pd.read_csv("books.csv")
book_features_v2 = pd.read_csv("book_features_v2.csv")

df_books = df_books.head(11)

book_features = np.zeros((len(df_books), 768*7)) 

def convert_to_vector(x):
    x = x.strip().rstrip("\\n")
    return np.fromstring(x[1:-1], sep=' ')

for i, row in df_books.iterrows():
    isbn = row['isbn']
    print(book_features_v2.loc[book_features_v2['isbn'] == isbn, 'feature'].apply(convert_to_vector))
    # book_features[i] = book_features_v2.loc[book_features_v2['isbn'] == isbn, 'feature']

# knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
# knn_model.fit(book_features)

# sample_features = convert_string_to_vector(book_features_v2.loc[book_features_v2['isbn'] == '0192126040', 'feature'].values)

# distances, indices = knn_model.kneighbors(sample_features.reshape(1, -1))

# for index in indices[0][1:]:
#     book_info = df_books.iloc[index]
#     print(book_info['isbn'])