import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from collections import Counter

def convert_to_vector(x):
    x = x.strip().rstrip("\\n")
    array = np.fromstring(x[1:-1], dtype=float, sep=' ')
    return array.astype(float)

def get_users_with_rating(isbn, rating_dataset):
    users_with_rating = rating_dataset.loc[rating_dataset['isbn'] == isbn, 'user-id']
    return users_with_rating

def predict_rating(user_id, isbn, rating_dataset, user_features_v2, most_common_rating):
    users_with_rating = get_users_with_rating(isbn, rating_dataset)
    user_features = user_features_v2.loc[user_features_v2['user-id'].isin(users_with_rating), 'feature'].apply(convert_to_vector)

    if user_features.empty:
        return most_common_rating

    input_user_features = convert_to_vector(user_features_v2.loc[user_features_v2['user-id'] == user_id, 'feature'].values[0])
    user_features_array = np.vstack(user_features.values)

    cosine_similarities = cosine_similarity([input_user_features], user_features_array)
    cosine_distances = 1.0 - cosine_similarities[0]

    ratings = rating_dataset.loc[rating_dataset['isbn'] == isbn, 'book-rating'].values

    distances = cosine_distances
    weights = 1 / (np.array(distances) + 0.0001)
    weighted_average_rating = np.sum(ratings * weights) / np.sum(weights)
    rounded_rating = int(np.round(weighted_average_rating))
    final_rating = max(1, min(rounded_rating, 5))
    return final_rating

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
            predicted_rating = predict_rating(user_id, isbn, train_data, user_features_v2, most_common_rating)
            true_ratings.append(true_rating)
            predicted_ratings.append(predicted_rating)

        precision = precision_score(true_ratings, predicted_ratings, average='micro')
        precisions.append(precision)
        recall = recall_score(true_ratings, predicted_ratings, average='micro')
        recalls.append(recall)

    average_precision = np.mean(precisions)
    print("Average Precision:", average_precision)
    average_recall = np.mean(recalls)
    print("Average Recall:", average_recall)

main()