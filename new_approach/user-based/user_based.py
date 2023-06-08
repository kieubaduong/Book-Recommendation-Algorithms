import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors


df_rating = pd.read_csv('../../dataset/processed_dataset/ratings.csv')
df_book = pd.read_csv('../../dataset/processed_dataset/books.csv')
df_user = pd.read_csv('../../dataset/processed_dataset/users.csv')
    
df_user = df_user.head(11)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
default_features = np.zeros((768,))

def extract_features(book_info, rating):
        title = book_info['title'].values[0]
        author = book_info['author'].values[0]
        publish_year = book_info['year'].values[0]
        publisher = book_info['publisher'].values[0]
        tags = book_info['tags'].values[0]
        description = book_info['description'].values[0]
        genres = book_info['genres'].values[0]

        max_length = 512 - len(title) - len(author) - 4 - len(publisher)
        tags = tags[:max_length]
        description = description[:max_length]
        genres = genres[:max_length]

        input_text = f"{title} {author} {publish_year} {publisher} {tags} {description} {genres}"
        inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length', max_length=512, return_tensors='pt')

        outputs = model(**inputs)

        hidden_state = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
        weighted_features = hidden_state * (rating - 3)
        return weighted_features

def extract_user_features(user_id):
    user_ratings = df_rating[df_rating['user-id'] == user_id]
    rated_books = user_ratings['isbn'].tolist()

    book_features = []
    
    if len(rated_books) == 0:
        return default_features
    else:
        for book_isbn in rated_books:
            rating_info = df_rating[(df_rating['user-id'] == user_id) & (df_rating['isbn'] == book_isbn)]
            if not rating_info.empty:
                rating = rating_info['book-rating'].values[0]
                book_info = df_book[df_book['isbn'] == book_isbn]
                if not book_info.empty:
                    features = extract_features(book_info, rating)
                    book_features.append(features)

        book_features_array = np.array(book_features)
        user_features = np.prod(book_features_array, axis=0)
        return user_features

user_features = []
user_ids = df_user['user-id'].tolist()

# Trích xuất đặc trưng cho từng người dùng và lưu vào user_features
for user_id in user_ids:
    features = extract_user_features(user_id)
    user_features.append(features)

# Chuyển user_features thành một ma trận numpy
user_features_array = np.array(user_features)

# Sử dụng knn để gom nhóm người dùng
n_neighbors = 10
knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
knn.fit(user_features_array)

# Lấy vector đặc trưng của người dùng đang xét
user_id = 276847
target_user_features = extract_user_features(user_id)

# Tìm 10 người dùng gần nhất với người dùng đang xét
distances, indices = knn.kneighbors(target_user_features.reshape(1, -1), n_neighbors=n_neighbors+1)

# Loại bỏ chính người dùng đang xét ra khỏi danh sách
indices = indices.flatten()[1:]

# Lấy thông tin về người dùng gần nhất
nearest_users = df_user.loc[indices]

recommended_books = []

for user_id in nearest_users:
    user_ratings = df_rating[(df_rating['user-id'] == user_id) & (df_rating['isbn'].isin(nearest_users))]
    liked_books = user_ratings[user_ratings['book-rating'] >= 3]['isbn'].tolist()
    unread_books = [book for book in liked_books if book not in nearest_users]
    recommended_books.extend(unread_books)
    if len(recommended_books) >= 10:
        break

for book in recommended_books:
    print(book)