import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('../../dataset/processed_dataset/books.csv')

def process_description(description):
    description = re.sub(r'[^a-zA-Z0-9\s]', '', description)
    description = description.lower()
    stop_words = set(stopwords.words('english'))
    description = ' '.join([word for word in description.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    description = ' '.join([lemmatizer.lemmatize(word) for word in description.split()])
    return description

df['description'] = df['description'].apply(process_description)

df.to_csv('../../dataset/processed_dataset/books.csv', index=False)