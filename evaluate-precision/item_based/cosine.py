import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import Counter

def calculate_weighted_average_rating(cosine_distances, user_ratings):
    distances = cosine_distances
    weights = 1 / (np.array(distances) + 0.0001)
    weighted_average_rating = np.sum(user_ratings * weights) / np.sum(weights)
    rounded_rating = int(np.round(weighted_average_rating))
    final_rating = max(1, min(rounded_rating, 5))
    return final_rating

def predictRating(pivot_table, user_id, isbn, rating_dataset, most_common_rating):
    other_ratings = rating_dataset[(rating_dataset['user-id'] == user_id) & (rating_dataset['isbn'] != isbn)]
    other_ratings = other_ratings.set_index('isbn')['book-rating']

    if other_ratings.empty:
        return most_common_rating

    target_column = pivot_table[isbn]
    other_columns = pivot_table.drop(columns=isbn)

    similarities = cosine_similarity(target_column.values.reshape(1, -1), other_columns.values.T)
    similarities_series = pd.Series(similarities[0], index=other_columns.columns)

    cosine_distances = []
    ratings = []

    for other_isbn, similarity in similarities_series.items():
        if other_isbn in other_ratings.index:
            cosine_distances.append(similarity)
            ratings.append(other_ratings[other_isbn])

    return calculate_weighted_average_rating(cosine_distances, ratings)

def main():
    rating_dataset = pd.read_csv("/content/drive/MyDrive/dataset/ratings.csv")
    rating_dataset = rating_dataset.head(1000)

    ratings = rating_dataset['book-rating'].values
    rating_counts = Counter(ratings)
    most_common_rating = rating_counts.most_common(1)[0][0]

    pivot_table = rating_dataset.pivot_table(index="user-id", columns="isbn", values="book-rating", fill_value=-1).astype(np.int8)

    precisions = []
    recalls = []
    for _ in range(10):
        train_data, test_data = train_test_split(rating_dataset, test_size=0.2)
        predicted_ratings = []
        for _, row in test_data.iterrows():
            user_id = row['user-id']
            isbn = row['isbn']
            predicted_rating = predictRating(pivot_table, user_id, isbn, train_data, most_common_rating)
            predicted_ratings.append(predicted_rating)
        true_ratings = test_data['book-rating'].tolist()

        recall = recall_score(true_ratings, predicted_ratings, average='weighted', zero_division=1)
        recalls.append(recall)
        precision = precision_score(true_ratings, predicted_ratings, average='weighted', zero_division=1)
        precisions.append(precision)

    average_precision = np.mean(precisions)
    print("Average Precision:", average_precision)
    average_recall = np.mean(recalls)
    print("Average Recall:", average_recall)

main()