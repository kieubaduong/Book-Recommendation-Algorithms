import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def calculate_weighted_average_rating(cosine_distances, user_ratings):
    distances = cosine_distances
    weights = 1 / (np.array(distances) + 0.0001)
    weighted_average_rating = np.sum(user_ratings * weights) / np.sum(weights)
    rounded_rating = int(np.round(weighted_average_rating))
    final_rating = max(1, min(rounded_rating, 5))
    return final_rating

def calculate_cosine_distances(target_column, other_columns):
    similarities = cosine_similarity(target_column.values.reshape(1, -1), other_columns.values)
    cosine_distances = 1 - similarities[0]
    return cosine_distances

def predictRatingKMeans(pivot_table, user_id, isbn, rating_dataset, most_common_rating, n_clusters=10):
    other_ratings = rating_dataset[(rating_dataset['user-id'] == user_id) & (rating_dataset['isbn'] != isbn)]
    other_ratings = other_ratings.set_index('isbn')['book-rating']

    if other_ratings.empty:
        return most_common_rating

    target_column = pivot_table[isbn]
    other_columns = pivot_table.drop(columns=isbn)

    valid_indices = pivot_table.columns[pivot_table.columns != -1]
    valid_indices = valid_indices.intersection(target_column.index)

    target_column_valid = target_column.loc[valid_indices]
    other_columns_valid = other_columns[valid_indices]

    if other_columns_valid.shape[1] == 0:
        return most_common_rating

    other_columns_array = other_columns_valid.values

    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(other_columns_array)

    target_cluster_label = kmeans.predict(target_column_valid.values.reshape(1, -1))[0]
    target_cluster_indices = np.where(cluster_labels == target_cluster_label)[0]

    target_cluster_columns = other_columns_valid.iloc[target_cluster_indices]
    cosine_distances = calculate_cosine_distances(target_column_valid, target_cluster_columns)

    ratings = other_ratings[target_cluster_columns.index]

    return calculate_weighted_average_rating(cosine_distances, ratings)

def main():
    rating_dataset = pd.read_csv("/content/drive/MyDrive/dataset/ratings.csv")

    ratings = rating_dataset['book-rating'].values
    rating_counts = Counter(ratings)
    most_common_rating = rating_counts.most_common(1)[0][0]

    pivot_table = rating_dataset.pivot_table(index="user-id", columns="isbn", values="book-rating", fill_value=-1).astype(np.int8)

    precisions_kmeans = []
    recalls_kmeans = []

    for _ in range(10):
        train_data, test_data = train_test_split(rating_dataset, test_size=0.2)
        predicted_ratings_kmeans = []

        for _, row in test_data.iterrows():
            user_id = row['user-id']
            isbn = row['isbn']

            predicted_rating_kmeans = predictRatingKMeans(pivot_table, user_id, isbn, train_data, most_common_rating)
            predicted_ratings_kmeans.append(predicted_rating_kmeans)

        true_ratings = test_data['book-rating'].tolist()

        recall_kmeans = recall_score(true_ratings, predicted_ratings_kmeans, average='weighted', zero_division=1)
        recalls_kmeans.append(recall_kmeans)
        precision_kmeans = precision_score(true_ratings, predicted_ratings_kmeans, average='weighted', zero_division=1)
        precisions_kmeans.append(precision_kmeans)

    average_precision_kmeans = np.mean(precisions_kmeans)
    print("Average Precision (KMeans):", average_precision_kmeans)
    average_recall_kmeans = np.mean(recalls_kmeans)
    print("Average Recall (KMeans):", average_recall_kmeans)

main()