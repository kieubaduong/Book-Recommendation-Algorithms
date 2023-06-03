import pandas as pd
import csv
import re

df = pd.read_csv('../new_dataset/crawled_books.csv')

columns_to_check = ['isbn', 'title', 'author', 'year', 'publisher', 'image_s', 'image_m', 'image_l', 'tags', 'description', 'genres']

df = df.dropna(subset=columns_to_check)

df.to_csv('processing_crawled_books.csv', index=False)