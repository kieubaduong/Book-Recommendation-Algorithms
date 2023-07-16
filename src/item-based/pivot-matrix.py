import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

def main(isbn):
  ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv', dtype={'book-rating': 'int8', 'user-id': 'int16'})

  pivot_table = ratings.pivot_table(index="user-id", columns="isbn", values="book-rating", fill_value=-1).astype(np.int8)
  
  print(pivot_table.head())
  target_column = pivot_table[isbn]

  other_columns = pivot_table.drop(columns=isbn)
  similarities = cosine_similarity(target_column.values.reshape(1, -1), other_columns.values.T)
  similarities_series = pd.Series(similarities[0], index=other_columns.columns)
  top_10_books = similarities_series.sort_values(ascending=False).head(10)

  book_list = []
  for id, _ in top_10_books.items():
    book_list.append(id)

  return book_list

# start_time = time.time()

isbn = '0002190915'
main(isbn)

# end_time = time.time()
# execution_time = end_time - start_time
# print("Thời gian thực thi:", execution_time, "giây")