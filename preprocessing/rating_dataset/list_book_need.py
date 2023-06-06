import pandas as pd

df_ratings = pd.read_csv('../../dataset/raw_dataset/Ratings.csv')


# Đọc dataset book vào dataframe
df_books = pd.read_csv('../../dataset/raw_dataset/Books.csv')

# Ghép hai dataframe dựa trên cột 'book-id'
df_merged = pd.merge(df_ratings, df_books, on='ISBN')

print(len(df_merged))