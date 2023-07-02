import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

start_time = time.time()

books = pd.read_csv('../../dataset/processed_dataset/books.csv')
ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

pivot_table = ratings.pivot_table(index="user-id", columns="isbn", values="book-rating", fill_value=-1)

isbn = '0002190915'  # Book ID của quyển sách đang xét

target_column = pivot_table[isbn]

# Lấy các cột còn lại trong pivot table
other_columns = pivot_table.drop(columns=isbn)

# Tính cosine similarity giữa target_column và các cột còn lại
similarities = cosine_similarity(target_column.values.reshape(1, -1), other_columns.values.T)

# Chuyển mảng similarities thành một Series với index là ISBN của các sách
similarities_series = pd.Series(similarities[0], index=other_columns.columns)

# Sắp xếp theo thứ tự giảm dần và lấy 10 sách có cosine similarity cao nhất
top_10_books = similarities_series.sort_values(ascending=False).head(10)

book_list = []
for isbn, similarity_score in top_10_books.items():
    book_list.append(isbn)

print(book_list)

end_time = time.time()
execution_time = end_time - start_time
print("Thời gian thực thi:", execution_time, "giây")