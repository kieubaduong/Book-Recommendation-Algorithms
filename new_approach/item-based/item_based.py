import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

books = pd.read_csv('../../dataset/processed_dataset/books.csv')
ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

def item_based_recommendation(book_id, books, ratings, top_n=10):
    # Lấy danh sách user đã rating quyển sách đang xét
    book_ratings = ratings[ratings['isbn'] == book_id]

    # Tạo vector rating của các user cho quyển sách
    users = ratings['user-id'].unique()
    ratings_vector = np.full(len(users), -1)  # Khởi tạo vector rating ban đầu với giá trị -1
    for index, row in book_ratings.iterrows():
        user_id = row['user-id']
        rating = row['book-rating']
        user_index = np.where(users == user_id)[0]  # Tìm vị trí của user trong vector
        ratings_vector[user_index] = rating

    # Tính khoảng cách cosine giữa vector rating của sách đang xét và các sách khác
    book_vectors = []
    for index, book_row in books.iterrows():
        book_ratings_vector = np.full(len(users), -1)
        book_ratings = ratings[ratings['isbn'] == book_row['isbn']]
        for index, row in book_ratings.iterrows():
            user_id = row['user-id']
            rating = row['book-rating']
            user_index = np.where(users == user_id)[0]
            book_ratings_vector[user_index] = rating
        book_vectors.append(book_ratings_vector)

    # Tính ma trận khoảng cách cosine giữa các vector rating
    distances = cosine_distances([ratings_vector], book_vectors)[0]

    # Lấy ra top N quyển sách gần nhất
    top_indices = np.argsort(distances)[:top_n]
    top_books = books.iloc[top_indices]

    return top_books

book_id = '0393037355'  # Book ID của quyển sách đang xét
top_books = item_based_recommendation(book_id, books, ratings, top_n=10)
print(top_books)
