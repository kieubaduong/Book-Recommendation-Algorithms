import pandas as pd

df_books = pd.read_csv('../../dataset/processed_dataset/books.csv')

df_ratings = pd.read_csv('../../dataset/processed_dataset/Ratings.csv')

# Ghép hai dataframe dựa trên cột 'isbn'
df_merged = pd.merge(df_ratings, df_books, on='isbn', how='inner')

# Lấy danh sách các isbn có trong book dataset
isbn_in_books = df_books['isbn'].unique()

# Lọc các rating trong rating dataset dựa trên isbn có trong book dataset
df_filtered_ratings = df_ratings[df_ratings['isbn'].isin(isbn_in_books)]

df_filtered_ratings.to_csv('../../dataset/processed_dataset/ratings.csv', index=False)
