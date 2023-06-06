import pandas as pd

# Đọc dataset rating và book vào các dataframe
df_book = pd.read_csv('../dataset/processed_dataset/books.csv')
df_rating = pd.read_csv('../dataset/processed_dataset/ratings.csv')

# Lấy danh sách tất cả các ISBN trong dataset rating
isbn_rating = df_rating['isbn'].unique()

# Lấy danh sách tất cả các ISBN trong dataset book
isbn_book = df_book['isbn'].unique()

# Kiểm tra xem tất cả các ISBN trong dataset rating có tồn tại trong dataset book hay không
missing_isbn = set(isbn_rating) - set(isbn_book)

# In ra các ISBN không tồn tại trong dataset book
for isbn in missing_isbn:
    print(f"ISBN {isbn} không tồn tại trong dataset book")
