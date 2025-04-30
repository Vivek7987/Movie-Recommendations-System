import pickle
import streamlit as st
import requests

# --- Load the pickle files (ensure these files are in the same folder as this script) ---
try:
    with open("movie_list.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# --- Poster fetch function ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else ""
    return full_path

# --- Recommendation logic ---
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters

# --- Streamlit UI ---
st.header('🎬 Movie Recommender System')
movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(recommended_movie_names[i])
            st.image(recommended_movie_posters[i])
