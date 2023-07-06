import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import csv
import os

books = pd.read_csv("/content/drive/MyDrive/Dataset/processed/books.csv")
ratings = pd.read_csv("/content/drive/MyDrive/Dataset/processed/ratings.csv")

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

def save_recommendations_to_csv(df_books, book_features, output_path):
    exist_books = []
    if os.path.isfile(output_path):
        exist_df = pd.read_csv(output_path)
        exist_books = exist_df['isbn'].tolist()

    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if not exist_books:
            writer.writerow(['isbn', 'top-10'])
        for _, book_row in df_books.iterrows():
            book_id = book_row['isbn']
            if book_id in exist_books:
                continue
            top_books = item_based_recommendation(book_id, book_features, top_n=10)
            top_isbns = ';'.join(top_books['isbn'])
            writer.writerow([book_id, top_isbns])


book_features = extract_book_features(books, ratings)
output_path = "/content/drive/MyDrive/Dataset/featured/item_based.csv"
save_recommendations_to_csv(books, book_features, output_path)
print(f"Finished writing recommendations to {output_path}.")
