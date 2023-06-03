import pandas as pd

df = pd.read_csv('../dataset/crawled_dataset/books.csv', encoding='utf-8')

columns_to_check = ['isbn', 'title', 'author', 'year', 'publisher', 'image_s', 'image_m', 'image_l', 'tags', 'description', 'genres']

df.replace('', float('NaN'), inplace=True)
df = df.dropna(subset=columns_to_check)

df.reset_index(drop=True, inplace=True)

df.to_csv('../dataset/crawled_dataset/books.csv', index=False)