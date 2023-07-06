import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import tracemalloc
import platform

start_time = time.time()
tracemalloc.start()
architecture = platform.architecture()[0]

print(f"Python is running as {architecture}-bit")

# ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv', dtype={'book-rating': 'int8', 'user-id': 'int16'})

# pivot_table = ratings.pivot_table(index="user-id", columns="isbn", values="book-rating", fill_value=-1).astype(np.int8)
pivot_table = pd.read_csv('pivot_table.csv', dtype=np.int8)

isbn = '0002190915'  # Book ID của quyển sách đang xét

target_column = pivot_table[isbn]

# Lấy các cột còn lại trong pivot table
other_columns = pivot_table.drop(columns=isbn)

# print(other_columns.values.T.info())

# # Tính cosine similarity giữa target_column và các cột còn lại
similarities = cosine_similarity(target_column.values.reshape(1, -1), other_columns.values.T)
# print(similarities.nbytes)


# # Chuyển mảng similarities thành một Series với index là ISBN của các sách
# similarities_series = pd.Series(similarities[0], index=other_columns.columns, dtype=np.int8)
# print(similarities_series.info())

# # Sắp xếp theo thứ tự giảm dần và lấy 10 sách có cosine similarity cao nhất
# top_10_books = similarities_series.sort_values(ascending=False).head(10)

# book_list = []
# for isbn, similarity_score in top_10_books.items():
#     book_list.append(isbn)

# # print(book_list)

# print(pivot_table.info())

# end_time = time.time()
# execution_time = end_time - start_time
# print("Thời gian thực thi:", execution_time, "giây")

snapshot = tracemalloc.take_snapshot()

# Lấy thông tin về việc sử dụng bộ nhớ
top_stats = snapshot.statistics('lineno')

# Tính tổng dung lượng bộ nhớ
total_memory = sum(stat.size for stat in top_stats)

# In ra tổng dung lượng bộ nhớ
print(f'Total memory usage: {total_memory} bytes')