import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


df = pd.read_csv('../dataset/processed_dataset/books.csv')

stop_words = set(nltk.corpus.stopwords.words('english'))

def process_tags(row):
    tags = row['tags']
    tag_list = tags.split(';')
    processed_tags = [re.sub(r'[^a-zA-Z]', '', word.strip()) for word in tag_list if re.sub(r'[^a-zA-Z]', '', word.strip()).lower() not in stop_words]
    processed_tags = ';'.join(processed_tags)
    return processed_tags

df['tags'] = df.apply(process_tags, axis=1)

df['tags'] = df['tags'].apply(lambda x: re.sub(r';+', ';', str(x)) if isinstance(x, str) else x)
df['tags'] = df['tags'].apply(lambda x: re.sub(r';$', '', str(x)) if isinstance(x, str) else x)

df.to_csv('../dataset/processed_dataset/books.csv', index=False)