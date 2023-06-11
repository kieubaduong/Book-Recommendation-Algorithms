
import pandas as pd

df_rating = pd.read_csv('../dataset/processed_dataset/ratings.csv')
df_book = pd.read_csv('../dataset/processed_dataset/books.csv')
df_user = pd.read_csv('../dataset/processed_dataset/users.csv')


# Kết hợp DataFrame df_book và df_rating dựa trên cột 'isbn'
merged_df = df_book.merge(df_rating, on='isbn', how='left')

# Lọc các dòng trong df_book không có tương ứng trong df_rating
books_without_rating = merged_df[merged_df['user-id'].isnull()]

# Tạo một mảng boolean cho biết các quyển sách có trong books_without_rating hay không
is_not_rated = df_book['isbn'].isin(books_without_rating['isbn'])

# Lọc và loại bỏ các quyển sách không được rating từ DataFrame df_book
df_book_rated = df_book[~is_not_rated]

# Hiển thị DataFrame df_book sau khi loại bỏ các quyển sách không được rating
# df_book_rated.to_csv('../dataset/processed_dataset/books.csv', index=False)
