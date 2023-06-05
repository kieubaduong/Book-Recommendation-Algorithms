import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel

df_books = pd.read_csv("../../dataset/processed_dataset/books.csv")

df_books_subset = df_books.head(100)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(row):
    try:
        title = row['title']
        author = row['author']
        publish_year = row['year']
        publisher = row['publisher']
        tags = row['tags']
        description = row['description']
        genres = row['genres']

        input_text = f"{title} {author} {publish_year} {publisher} {tags} {description} {genres}"
        inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length', max_length=128, return_tensors='pt')

        outputs = model(**inputs)

        hidden_state = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
        
        return hidden_state
    except:
        print(row)
    return np.zeros(768)
    


book_features = np.zeros((len(df_books_subset), 768))

for i, row in df_books_subset.iterrows():
    book_features[i] = extract_features(row)

# knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
# knn_model.fit(book_features)

# sample_book = {'title': 'Slumdog Millionaire', 'author': 'Vikas Swarup', 'year': '2008', 'publisher': 'HarperCollins', 'tags': 'India', 'description': 'The story of a nobody who became a somebody.', 'genres': 'Fiction'}
# sample_features = extract_features(sample_book)

# print(sample_features.shape)

# distances, indices = knn_model.kneighbors(sample_features.reshape(1, -1))   

# print("distance" + str(distances))

# for index in indices:
#     book_info = df_books_subset.iloc[index]
#     print(book_info)