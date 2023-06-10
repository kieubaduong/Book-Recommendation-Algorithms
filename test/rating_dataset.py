import pandas as pd


df_rating = pd.read_csv('../dataset/processed_dataset/ratings.csv')
df_book = pd.read_csv('../dataset/processed_dataset/books.csv')

missing_isbn = df_rating[~df_rating['isbn'].isin(df_book['isbn'])]['isbn'].unique()

for isbn in missing_isbn:
    print(isbn)

