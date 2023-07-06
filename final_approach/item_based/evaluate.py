import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def calculate_weighted_average_rating(cosine_distances, user_ratings):
    distances = cosine_distances
    weights = 1 / (np.array(distances) + 0.0001)
    weighted_average_rating = np.sum(user_ratings * weights) / np.sum(weights)
    rounded_rating = int(np.round(weighted_average_rating))
    final_rating = max(1, min(rounded_rating, 5))
    return final_rating

def predictRating(pivot_table, user_id, isbn, rating_dataset):
    

    other_ratings = rating_dataset[(rating_dataset['user-id'] == user_id) & (rating_dataset['isbn'] != isbn)]
    other_ratings = other_ratings.set_index('isbn')['book-rating']

    if other_ratings.empty:
        return 3

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

def evaluate_accuracy(rating_dataset):
    train_data, test_data = train_test_split(rating_dataset, test_size=0.2)
    pivot_table = rating_dataset.pivot_table(index="user-id", columns="isbn", values="book-rating", fill_value=-1).astype(np.int8)

    predicted_ratings = []
    for _, row in test_data.iterrows():
        user_id = row['user-id']
        isbn = row['isbn']
        predicted_rating = predictRating(pivot_table, user_id, isbn, train_data)
        predicted_ratings.append(predicted_rating)

    true_ratings = test_data['book-rating'].tolist()
    accuracy = accuracy_score(true_ratings, predicted_ratings)
    return accuracy

def main():
    rating_dataset = pd.read_csv("ratings.csv")

    # rating_dataset = rating_dataset.head(10000)

    accuracy = evaluate_accuracy(rating_dataset)
    print("Accuracy:", accuracy)

main()

