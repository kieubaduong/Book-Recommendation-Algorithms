import pandas as pd

df = pd.read_csv('../new_dataset/processed_crawled_books.csv')

genres_list = []

for index, row in df.iterrows():
    genres = row['genres'].split(';')
    genres_list.extend(genres)
    genres_list = [genre.strip() for genre in genres_list]

unique_genres = list(set(genres_list))


pd.DataFrame(unique_genres, columns=['genre']).to_csv('genres.csv', index=False)
    
    