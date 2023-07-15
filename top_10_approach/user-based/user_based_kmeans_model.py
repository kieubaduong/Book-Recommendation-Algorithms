import numpy as np
import pandas as pd
import json
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=DeprecationWarning)

def extract_user_features(ratings, df_book, book_features_v2):
    ratings = json.loads(ratings)
    def convert_to_vector(x):
        x = x.strip().rstrip("\\n")
        return np.fromstring(x[1:-1], sep=' ')

    book_features_v2['feature'] = book_features_v2['feature'].apply(convert_to_vector)

    book_features = []
    default_features = np.zeros((768,))

    if not ratings:
        return default_features

    for isbn, rating in ratings.items():
        book_info = df_book[df_book['isbn'] == isbn]
        if not book_info.empty:
            features = book_features_v2[book_features_v2['isbn'] == isbn]['feature'].values[0] * (rating - 3)
            book_features.append(features)

    if not book_features:
        return default_features

    book_features_array = np.array(book_features)
    user_features = np.prod(book_features_array, axis=0)
    return user_features

def get_recommended_books(input_vector, df_user_features, df_rating, df_book):
    df_user_features = df_user_features.drop_duplicates(subset='feature', keep='first')
    user_features = np.array([np.fromstring(features[1:-1], dtype=float, sep=' ') for features in df_user_features['feature']])
    user_ids = df_user_features['user-id'].values
    
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(user_features)
    nearest_cluster = kmeans.predict([input_vector])[0]
    
    users_in_cluster = user_ids[cluster_labels == nearest_cluster]
    users_in_cluster_features = user_features[cluster_labels == nearest_cluster]

    k = 10
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(users_in_cluster_features)
    distances, indices = nn.kneighbors([input_vector])

    nearest_users = users_in_cluster[indices[0]]
    recommended_books = []
    book_seen = set()

    for user_id in nearest_users:
        user_ratings = df_rating.loc[(df_rating['user-id'] == user_id) & (~df_rating['isbn'].isin(book_seen)), 'isbn']
        liked_books = user_ratings.tolist()

        for book_id in liked_books:
            if book_id not in book_seen and len(recommended_books) < 10:
                recommended_books.append(book_id)
                book_seen.add(book_id)

        if len(recommended_books) >= 10:
            break

    recommended_books_info = []
    for book_id in recommended_books:
        book_info = df_book[df_book['isbn'] == book_id]
        if not book_info.empty:
            book_info_dict = book_info.iloc[0]['isbn']
            recommended_books_info.append(book_info_dict)

    print(recommended_books_info)
    return recommended_books_info

def main(input_data):
    book_features_v2 = pd.read_csv("book_features_v2.csv")
    df_book = pd.read_csv("books.csv")
    df_rating = pd.read_csv("ratings.csv")
    df_user_features = pd.read_csv("user_features_v2.csv")
    
    output = extract_user_features(input_data, df_book, book_features_v2)
    output = get_recommended_books(output, df_user_features, df_rating, df_book)

    return json.dumps(output)

main("{\"1564743861\": 3, \"0064410889\": 4,\"0060561580\": 3, \"0060561572\": 3,\"0060537264\": 2,\"1879505622\": 3,\"1578565480\": 3,\"1582380805\": 1,\"0060987103\": 5,\"0060973129\": 5,\"0192126040\": 5,\"0345260317\": 1,\"042511774X\": 1}")