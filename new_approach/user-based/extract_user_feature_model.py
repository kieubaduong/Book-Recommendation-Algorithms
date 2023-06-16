import numpy as np
import pandas as pd

default_features = np.zeros((768,))
book_features_v2 = pd.read_csv("/content/drive/MyDrive/Dataset/featured/book_features_v2.csv")
df_book = pd.read_csv("/content/drive/MyDrive/Dataset/processed/books.csv")

def convert_to_vector(x):
    x = x.strip().rstrip("\\n")
    return np.fromstring(x[1:-1], sep=' ')
    
book_features_v2['feature'] = book_features_v2['feature'].apply(convert_to_vector)  

def extract_user_features(ratings):
    book_features = []
    
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

# Đầu vào: ratings là một từ điển (dictionary) với key là isbn và value là rating của quyển sách
ratings = {
    '1573222011': 1,
    '0375708014': 4,
    '0375719156': 2,
    '0517546779' : 2,
    # ...Thêm các quyển sách và rating tương ứng tại đây...
}

# Gọi hàm extract_user_features với ratings là đầu vào
user_features = extract_user_features(ratings)

# In ra kết quả user_features
print(user_features)
