import pandas as pd

# Đọc dữ liệu từ file book_features.csv và book.csv
df_book_features = pd.read_csv("../../dataset/book_features.csv")
df_book = pd.read_csv('../../dataset/processed_dataset/books.csv')

print(df_book_features.info())

# Lấy danh sách các giá trị isbn từ cả hai dataset
# isbn_book_features = set(df_book_features["isbn"])
# isbn_book = set(df_book["isbn"])

# # Tìm các giá trị isbn có trong book_features nhưng không có trong book
# missing_isbns = isbn_book_features - isbn_book

# # Loại bỏ các bản ghi có giá trị isbn không có trong book dataset
# df_book_features_filtered = df_book_features[~df_book_features["isbn"].isin(missing_isbns)]

# # In kết quả
# # df_book_features_filtered.to_csv("../../dataset/book_features.csv", index=False)

# print(df_book_features_filtered.info())