import pandas as pd

df_books = pd.read_csv('../../dataset/crawled_dataset/books.csv')

# Đọc dataset rating vào dataframe
df_ratings = pd.read_csv('../../dataset/raw_dataset/Ratings.csv')

# Ghép hai dataframe dựa trên cột 'book-id'
df_merged = pd.merge(df_books, df_ratings, on='isbn', how='inner')

# Loại bỏ các dòng có giá trị null (không có rating)
df_merged = df_merged.dropna(subset=['book-rating'])

# Lấy danh sách các book-id có trong rating dataset
book_ids_in_ratings = df_merged['isbn'].unique()

# Lọc các quyển sách trong book dataset dựa trên book-id có trong rating dataset
df_filtered_books = df_books[df_books['isbn'].isin(book_ids_in_ratings)]

df_filtered_books.to_csv('../../dataset/processed_dataset/books.csv', index=False)
