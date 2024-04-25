from google.colab import files
upload=files.upload()

from google.colab import files
upload=files.upload()

import numpy as np
import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies.head()

credits.head()

movies.shape

credits.shape

movies = movies.merge(credits, on='title')

movies.shape

movies.head(1)

movies.info()

movies=movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.head()

"""Data Preprocessing"""

movies.isnull().sum()

movies.dropna(inplace=True)

movies.duplicated().sum()

movies.iloc[0].genres

import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
      L.append(i['name'])
    return L

movies['genres']= movies['genres'].apply(convert)

movies.head()

movies.iloc[0].keywords

movies['keywords']

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
      if counter != 3:
        L.append(i['name'])
        counter += 1
      else:
        break
    return L

movies['cast']= movies['cast'].apply(convert3)

movies.head()

def director(obj):
    L = []
    for i in ast.literal_eval(obj):
      if i['job'] == 'Director':
        L.append(i['name'])
        break
    return L

movies['crew']= movies['crew'].apply(director)

movies.head()

movies['overview']= movies['overview'].apply(lambda x:x.split())

movies.head()

#Merging columns into one tag
#Transformation is done for genres, keywords, cast, and crew to remove spaces between them
# This is important as first and second name will be taken as different. And if we only take first name string, the model will get confused as many entity have same first name.

movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies.head()

movies['tags']= movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df= movies[['movie_id', 'title', 'tags']]
new_df

new_df.iloc[0].tags

new_df['tags']= new_df['tags'].apply(lambda x:" ".join(x))

new_df.head()

new_df['tags'][0]

#Converting this in lowercase
new_df['tags']= new_df['tags'].apply(lambda x:x.lower())

new_df.head()

"""Text vectorization"""

#Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 5000, stop_words='english')

vectors= cv.fit_transform(new_df['tags']).toarray()

vectors

vectors.shape

vectors[0]

"""Stemming"""

#Stemming is done so that actor and actors are considered same

import nltk
from nltk.stem.porter import PorterStemmer

ps= PorterStemmer()

def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

#Example
ps.stem('actors')

new_df['tags']= new_df['tags'].apply(stem)

#Calculating cosine distance between movies

from sklearn.metrics.pairwise import cosine_similarity

similarity= cosine_similarity(vectors)   #diagonal is always 1 and the similarity of movie m is calculated with movie m

sorted(list(enumerate(similarity[0])), reverse=True,key=lambda x:x[1])[1:6]  #To make sure the diagonal index is 1 after sorting

similarity.shape

#Recommendation Function
def recommend(movie):
  movie_index= new_df[new_df['title'] == movie].index[0]  #Fetching index
  distances= similarity[movie_index]
  movies_list= sorted(list(enumerate(distances)), reverse=True,key=lambda x:x[1])[1:6]

  for i in movies_list:
    print(new_df.iloc[i[0]].title)

recommend('Batman Begins')

new_df.iloc[260].title

import pickle

pickle.dump(new_df, open('movies.pkl','wb'))

new_df.to_dict()

pickle.dump(new_df.to_dict(), open('movie_dict.pkl','wb'))

pickle.dump(similarity, open('similarity.pkl', 'wb'))