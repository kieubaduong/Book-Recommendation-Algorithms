import pandas as pd
import numpy as np

df_books = pd.read_csv("../new_dataset/processed_books.csv", delimiter=",", usecols = [0,1,2,3,4], dtype={'Year-Of-Publication':object})


df_books['Year-Of-Publication'] = pd.to_numeric(df_books['Year-Of-Publication'],errors='coerce')

df_books = df_books.dropna()
df_books['Year-Of-Publication'] = df_books['Year-Of-Publication'].astype(int)

df_books_imputed = df_books.copy()

df_books_imputed.loc[df_books_imputed['Year-Of-Publication'] < 1000, 'Year-Of-Publication'] = np.nan

year_mean = df_books_imputed['Year-Of-Publication'].mode()[0]
df_books_imputed['Year-Of-Publication'].fillna(year_mean, inplace=True)

df_books = df_books_imputed

