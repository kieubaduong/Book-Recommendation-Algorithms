import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import time

books = pd.read_csv('../../dataset/processed_dataset/books.csv')
ratings = pd.read_csv('../../dataset/processed_dataset/ratings.csv')

# Trích xuất đặc trưng của tất cả quyển sách
def extract_book_features(books, ratings):
    users = ratings['user-id'].unique()
    book_features = []
    for _, book_row in books.iterrows():
        book_ratings_vector = np.full(len(users), -1)
        book_ratings = ratings[ratings['isbn'] == book_row['isbn']]
        for _, row in book_ratings.iterrows():
            user_id = row['user-id']
            rating = row['book-rating']
            user_index = np.where(users == user_id)[0]
            book_ratings_vector[user_index] = rating
        book_features.append(book_ratings_vector)
    return book_features

def item_based_recommendation(book_id, book_features, top_n=10):
    # Lấy vector rating của sách đang xét
    book_index = books[books['isbn'] == book_id].index[0]
    ratings_vector = book_features[book_index]

    # Tính khoảng cách cosine giữa vector rating của sách đang xét và các sách khác
    distances = cosine_distances([ratings_vector], book_features)[0]

    # Lấy ra top N quyển sách gần nhất (không chứa sách đang xét)
    top_indices = np.argsort(distances)[1:top_n+1]
    top_books = books.iloc[top_indices]

    return top_books

def main(book_id):
    book_features = extract_book_features(books, ratings)
    top_books = item_based_recommendation(book_id, book_features, top_n=10)
    print(top_books)
        
start_time = time.time()

book_id = '0393037355'  # Book ID của quyển sách đang xét
main(book_id)

end_time = time.time()
execution_time = end_time - start_time
print("Thời gian thực thi:", execution_time, "giây")