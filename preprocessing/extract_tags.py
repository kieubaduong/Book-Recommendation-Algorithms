import pandas as pd
import numpy as np

df = pd.read_csv('../new_dataset/processed_crawled_books.csv')

tags_array = np.concatenate(df['tags'].str.split(';').values)

unique_tags = np.unique(tags_array).tolist()

tags_df = pd.DataFrame({'tag': unique_tags})

tags_df.to_csv('tags.csv', index=False)
