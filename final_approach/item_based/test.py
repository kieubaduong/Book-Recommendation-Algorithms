import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def main(user_id, isbn):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, 'pivot_table.csv')
    pivot_table = pd.read_csv(csv_file_path, dtype=np.int8)

    # Đọc dữ liệu từ file rating dataset
    rating_file_path = os.path.join(current_dir, 'ratings.csv')
    rating_dataset = pd.read_csv(rating_file_path)

    target_column = pivot_table[isbn]
    other_columns = pivot_table.drop(columns=isbn)

    similarities = cosine_similarity(target_column.values.reshape(1, -1), other_columns.values.T)
    similarities_series = pd.Series(similarities[0], index=other_columns.columns)

    # Tạo danh sách các cặp (khoảng cách cosine, mã ISBN)
    cosine_distances = [(similarity, other_isbn) for other_isbn, similarity in similarities_series.items()]

    # Sắp xếp theo khoảng cách cosine
    cosine_distances.sort(key=lambda x: x[0])

    # In ra 100 khoảng cách đầu tiên
    print(f"First 100 cosine distances between book {isbn} and all other books:")
    for i, (distance, other_isbn) in enumerate(cosine_distances[:100], 1):
        print(f"{i}. ISBN: {other_isbn}, Cosine Distance: {distance}")

isbn = '0142001740'
user_id = 277157
main(user_id, isbn)
