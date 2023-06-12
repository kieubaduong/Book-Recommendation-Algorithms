import numpy as np
import csv
import pandas as pd
from transformers import BertTokenizer, BertModel

df_rating = pd.read_csv('../../dataset/processed_dataset/ratings.csv')
df_book = pd.read_csv('../../dataset/processed_dataset/books.csv')
df_user = pd.read_csv('../../dataset/processed_dataset/users.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_features(book_info):
    try:
      title = book_info['title']
      author = book_info['author']
      publish_year = book_info['year']
      publisher = book_info['publisher']
      tags = book_info['tags']
      description = book_info['description']
      genres = book_info['genres']

      max_length = 512 - len(title) - len(author) - 4 - len(publisher)
      tags = tags[:max_length]
      description = description[:max_length]
      genres = genres[:max_length]

      input_text = f"{title} {author} {publish_year} {publisher} {tags} {description} {genres}"
      inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length', max_length=512, return_tensors='pt')
      
      outputs = model(**inputs)
      hidden_state = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
      return hidden_state
    except:
        return None

error_isbns = []

df_book_features = pd.read_csv("../../dataset/book_features.csv")
extracted_isbns = df_book_features['isbn'].tolist()

for _, book_info in df_book.iterrows():
    isbn = book_info['isbn']
    if isbn in extracted_isbns:
        continue
    else:
        features = extract_features(book_info)
        if features is not None:
            with open("../../dataset/book_features.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([isbn, features])
        else:
            error_isbns.append(isbn)

error_isbn_df = pd.DataFrame({'isbn': error_isbns})
error_isbn_df.to_csv("../../dataset/error_isbns.csv", index=False)