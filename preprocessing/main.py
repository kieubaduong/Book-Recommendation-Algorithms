import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


df = pd.read_csv('../dataset/crawled_dataset/books.csv', encoding='utf-8')

# df = df.drop_duplicates(subset=['isbn'], keep='first')

df.to_csv('../dataset/processed_dataset/books.csv', index=False)

duplicate_isbn = df['isbn'].value_counts()
duplicate_isbn = duplicate_isbn[duplicate_isbn > 1]

if len(duplicate_isbn) > 0:
    print("Có mã sách bị trùng.")
    print(duplicate_isbn)
else:
    print("Không có mã sách bị trùng.")

# stop_words = set(nltk.corpus.stopwords.words('english'))

# def process_tags(row):
#     tags = row['tags']
#     tag_list = tags.split(';')
#     processed_tags = [re.sub(r'[^a-zA-Z]', '', word.strip()) for word in tag_list if re.sub(r'[^a-zA-Z]', '', word.strip()).lower() not in stop_words]
#     processed_tags = ';'.join(processed_tags)
#     return processed_tags

# df['tags'] = df.apply(process_tags, axis=1)

# df.to_csv('processed_tags.csv', index=False)