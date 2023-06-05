import re
import pandas as pd

df = pd.read_csv('../dataset/processed_dataset/books.csv')

def remove_special_characters(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

df['title'] = df['title'].apply(remove_special_characters)
df['author'] = df['author'].apply(remove_special_characters)
df['publisher'] = df['publisher'].apply(remove_special_characters)

df.to_csv('../dataset/processed_dataset/books.csv', index=False)