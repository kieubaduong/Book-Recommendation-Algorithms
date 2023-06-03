import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

df = pd.read_csv('../dataset/crawled_dataset/books.csv', encoding='utf-8')

df = df.head(10)['description']

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

df = df.apply(preprocess_text)

df.to_csv('descriptions.csv', index=False)

# df.to_csv('../dataset/processed_dataset/books.csv', index=False)