import pandas as pd

df = pd.read_csv('../dataset/crawled_dataset/books.csv', encoding='utf-8')

df = df.drop_duplicates(subset=['isbn'], keep='first')

df.to_csv('../dataset/processed_dataset/books.csv', index=False)