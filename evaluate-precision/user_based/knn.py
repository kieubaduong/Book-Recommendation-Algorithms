import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

def convert_to_vector(x):
    x = x.strip().rstrip("\\n")
    array = np.fromstring(x[1:-1], dtype=float, sep=' ')
    return array.astype(float)

def get_users_with_rating(isbn, rating_dataset):
    users_with_rating = rating_dataset.loc[rating_dataset['isbn'] == isbn, 'user-id']
    return users_with_rating

def predict_rating_kmeans(user_id, isbn, rating_dataset, user_features_v2, most_common_rating):
    users_with_rating = get_users_with_rating(isbn, rating_dataset)
    user_features = user_features_v2.loc[user_features_v2['user-id'].isin(users_with_rating), 'feature'].apply(convert_to_vector)

    if user_features.empty:
        return most_common_rating

    input_user_features = convert_to_vector(user_features_v2.loc[user_features_v2['user-id'] == user_id, 'feature'].values[0])
    user_features_array = np.vstack(user_features.values)

    n_clusters = 2
    n_samples = min(n_clusters, len(user_features_array))

    kmeans = KMeans(n_clusters=n_samples, random_state=42)
    kmeans.fit(user_features_array)
    cluster_label = kmeans.predict([input_user_features])

    cluster_ratings = rating_dataset.loc[rating_dataset['isbn'] == isbn].copy()
    cluster_ratings['cluster'] = kmeans.predict(user_features_array)

    if len(cluster_ratings) == 0:
        return most_common_rating

    n_neighbors = min(n_samples, 3)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(cluster_ratings[['user-id', 'cluster']], cluster_ratings['book-rating'])
    predicted_rating = knn.predict([[user_id, cluster_label[0]]])[0]

    return predicted_rating

def main():
    rating_dataset = pd.read_csv("/content/drive/MyDrive/dataset/ratings.csv")
    user_features_v2 = pd.read_csv("/content/drive/MyDrive/dataset/featured/user_features_v2.csv")

    ratings = rating_dataset['book-rating'].values
    rating_counts = Counter(ratings)
    most_common_rating = rating_counts.most_common(1)[0][0]

    recalls = []
    precisions = []
    for _ in range(10):
        train_data, test_data = train_test_split(rating_dataset, test_size=0.2)

        true_ratings = []
        predicted_ratings = []

        for _, row in test_data.iterrows():
            user_id = row['user-id']
            isbn = row['isbn']
            true_rating = row['book-rating']
            predicted_rating = predict_rating_kmeans(user_id, isbn, train_data, user_features_v2, most_common_rating)
            true_ratings.append(true_rating)
            predicted_ratings.append(predicted_rating)

        precision = precision_score(true_ratings, predicted_ratings, average='weighted', zero_division=1)
        precisions.append(precision)
        recall = recall_score(true_ratings, predicted_ratings, average='weighted', zero_division=1)
        recalls.append(recall)

    average_precision = np.mean(precisions)
    print("Average Precision:", average_precision)
    average_recall = np.mean(recalls)
    print("Average Recall:", average_recall)

main()