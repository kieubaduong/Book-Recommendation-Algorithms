import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


df = pd.read_csv('../new_dataset/processed_crawled_books.csv')

stop_words = set(nltk.corpus.stopwords.words('english'))

def process_tags(row):
    tags = row['tags']
    tag_list = tags.split(';')
    processed_tags = [re.sub(r'[^a-zA-Z]', '', word.strip()) for word in tag_list if re.sub(r'[^a-zA-Z]', '', word.strip()).lower() not in stop_words]
    processed_tags = ';'.join(processed_tags)
    return processed_tags

df.apply(process_tags, axis=1).to_csv('processed_tags.csv', index=False)