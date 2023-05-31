import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import missingno as msno

df_books = pd.read_csv('archive/Books.csv', delimiter=",", usecols = [0,1,2,3,4])
df_Users = pd.read_csv("archive/Users.csv")
df_rating = pd.read_csv("archive/Ratings.csv")

try:
    df_books['Year-Of-Publication']  = df_books['Year-Of-Publication'].astype(int)
except Exception as e:
    print(e)


df_books['Year-Of-Publication'] = pd.to_numeric(df_books['Year-Of-Publication'],errors='coerce')

df_books = df_books.dropna()
df_books['Year-Of-Publication'] = df_books['Year-Of-Publication'].astype(int)


plt.bar(pd.DataFrame(df_books['Year-Of-Publication'].value_counts()).sort_index().index, np.array(df_books['Year-Of-Publication'].value_counts()), width = 1)

df_books_imputed = df_books.copy()
df_books_imputed.loc[df_books_imputed['Year-Of-Publication'] < 1000, 'Year-Of-Publication'] = np.nan

year_mean = df_books_imputed['Year-Of-Publication'].mode()[0]

df_books_imputed['Year-Of-Publication'].fillna(year_mean, inplace=True)
df_books = df_books_imputed

msno.matrix(df_books)

df_Users['Age'] = df_Users['Age'].astype('Int32')

msno.matrix(df_Users)

age_mean = df_Users['Age'].mean()
age_mean = math.floor(age_mean)
df_Users['Age'].fillna(age_mean, inplace=True)

msno.matrix(df_Users)

df_rating = df_rating[df_rating['ISBN'].isin(df_books['ISBN'])]

f = ['count','mean']

df_books_summary = df_rating.groupby('ISBN')['Book-Rating'].agg(f)
df_books_summary.index = df_books_summary.index.map(str)

drop_book_list = df_books_summary[df_books_summary['count'] < 10].index

df_cust_summary = df_rating.groupby('User-ID')['Book-Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)

drop_cust_list = df_cust_summary[df_cust_summary['count'] < 10].index


df_rating = df_rating[~df_rating['ISBN'].isin(drop_book_list)]
df_rating = df_rating[~df_rating['User-ID'].isin(drop_cust_list)]


final_dataset = df_rating.pivot(index='ISBN',columns='User-ID',values='Book-Rating')


final_dataset = final_dataset.fillna(0)


final_dataset_csr = sparse.csr_matrix(final_dataset)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(final_dataset_csr)


def get_book_recommendation(ISBN, n_books_to_reccomend):
    book_list = df_books[df_books['ISBN']== ISBN]  
    
    if len(book_list):        
        book_idx = book_list.iloc[0]['ISBN']
        book_idx = final_dataset[final_dataset['ISBN'] == book_idx].index[0]
        distances , indices = knn.kneighbors(final_dataset_csr[book_idx], n_neighbors=n_books_to_reccomend+1)
        rec_book_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_book_indices:
            book_idx = final_dataset.iloc[val[0]]['ISBN']
            idx = df_books[df_books['ISBN'] == book_idx].index
            recommend_frame.append({'ISBN':df_books.iloc[idx]['ISBN'].values[0],
                                    'Book-Title':df_books.iloc[idx]['Book-Title'].values[0],
                                    'Book-Author':df_books.iloc[idx]['Book-Author'].values[0],
                                    'Year-Of-Publication':df_books.iloc[idx]['Year-Of-Publication'].values[0],
                                    'Publisher':df_books.iloc[idx]['Publisher'].values[0],
                                    'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_books_to_reccomend+1))
        return df
    else:
        return "Book not found. Re-check the ISBN"


def recommend_book(userID, n_books_to_reccomend = 10):
    ISBN = df_rating.loc[df_rating[df_rating['User-ID'] == userID]['Book-Rating'].idxmax()]['ISBN']
    recommendation = get_book_recommendation(ISBN, n_books_to_reccomend)
    return recommendation

recommend_book(8, n_books_to_reccomend = 10)