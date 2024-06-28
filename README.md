# Movie Recommender System
## Description
This project utilizes the TMDB 50000 dataset from Kaggle to create a content-based movie recommender system. By merging movie and crew data, we perform feature extraction and use Natural Language Processing (NLP) and cosine similarity
to recommend five similar movies based on a user's favorite movie. The GUI is built with Streamlit, and the TMDB API is used to fetch movie posters and additional information.    
## Acknowledgments
-  Special thanks to Shradha Dubey Maam
-  Special Thank to Nitish Singh(CampusX)
-  Kaggle for providing the TMDB 50000 dataset.
-  TMDB API for movie information and posters.
-  Streamlit for the interactive GUI framework.

## API Reference
-  [TMDB API](https://developer.themoviedb.org/reference/movie-details) for fetching movie posters and details.

# Appendix
## Data Loading and Preparation
1.  Loading Data:

-  The project utilizes the tmdb_5000_movies.csv and tmdb_5000_credits.csv datasets from Kaggle.
-  These datasets are loaded using pandas
  
```bash
  import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```
2.  Merging Data:

-  The movie and crew data are merged based on the movie title to create a comprehensive dataset.
  ```bash
     movies = movies.merge(credits, on='title')
  ```
3.  Feature Selection:

Essential features such as movie_id, title, overview, genres, keywords, cast, and crew are selected for further processing.
``` bash  
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```
4.  Data Cleaning:

-  Handling missing values and duplicates to ensure data integrity.
  ```bash 
 movies.dropna(inplace=True)
```
### Feature Extraction and Processing
1.  Converting JSON Strings to Lists:

-  The genres, keywords, cast, and crew fields are converted from JSON strings to Python lists for easier manipulation.
  ```bashimport ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
```
2.  Extracting Director and Top Cast:

-  Extracting the director's name and top three cast members from the crew and cast fields.
```bash 
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def fetch_top_cast(text):
    L = []
    count = 0
    for i in ast.literal_eval(text):
        if count < 3:
            L.append(i['name'])
        count += 1
    return L

movies['director'] = movies['crew'].apply(fetch_director)
movies['cast'] = movies['cast'].apply(fetch_top_cast)
```

### Model Building
1.  Creating Tags:

-  Combining overview, genres, keywords, cast, and director into a single tags field for model input.
  ``` bash
     movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']
  ```
2.  Text Preprocessing:

-   Converting text to lowercase and applying stemming to standardize the text data.
```bash  from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies['tags'] = movies['tags'].apply(stem)
```
3.  Vectorization and Similarity Calculation:

-  Converting text data into vectors using TF-IDF Vectorizer and calculating cosine similarity to find similar movies.
  ```bash from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```
### Saving and Loading Model
1.  Saving the Model:

-  The cosine similarity matrix and TF-IDF Vectorizer are saved as pickle files for later use in the Streamlit app.
  ```bash import pickle

pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```
2.  Loading the Model:

-  In the Streamlit app, the saved pickle files are loaded to provide movie recommendations
  ```bash  movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
```
### Streamlit App
1.  Building the GUI:

-  The Streamlit app allows users to input their favorite movie and receive five similar movie recommendations.
-  The TMDB API is used to fetch movie posters and additional details
  ```bash  import streamlit as st
  import requests

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append((movies.iloc[i[0]].title, fetch_poster(movie_id)))
    return recommended_movies

st.title('Movie Recommender System')
selected_movie_name = st.selectbox('Type or select a movie from the dropdown', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for title, poster in recommendations:
        st.image(poster)
        st.write(title)
```
## Authors
-  [Vivek Pal](https://www.linkedin.com/in/vivekpal798/)
## Demo
![Screenshot 2024-02-04 001906](https://github.com/Vivek7987/Movie-Recommendations-System/assets/111482462/13d42e4a-02b2-4af3-9330-e9c319931cc2)

## Deployment
Instructions for deploying the Streamlit app:

1.  Clone the repository
2.  Install the required packages
3.  Run the Streamlit app
#   Documentation
Additional documentation is provided in the Jupyter Notebook and comments within the code.

## Environment Variables
Set the following environment variables for the TMDB API:

-  TMDB_API_KEY
  
## Features
-  Movie recommendations based on content
-  Interactive GUI with Streamlit
-  Fetches movie posters and details using TMDB API

Github Profile - Skills
-  Data Structures and Algorithms
-  Web Devlopment
-  Machine Learning
-  Programming Languages
## Installation
-  Clone the repo
  ```bash
     git clone https://github.com/your-username/movie-recommender-system.git
 ```
## License
This project is licensed under the MIT License

# Tech
-  Python
-  Streamlit
-  TMDB API
-  NLP
-  Cosine Similarity
