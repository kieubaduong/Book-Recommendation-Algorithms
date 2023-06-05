import pandas as pd
import re

df = pd.read_csv('../../dataset/processed_dataset/users.csv')

def remove_special_characters(text):
    pattern = r'[^\w\s]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

for index, row in df.iterrows():
    location = row['location']
    cleaned_location = remove_special_characters(location)
    df.at[index, 'location'] = cleaned_location

df.to_csv('../../dataset/processed_dataset/users.csv', index=False)