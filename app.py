import streamlit as st
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = pickle.load(open('anime.pkl','rb'))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['genre'])

def generate_recommendations(anime_name, data, tfidf_matrix):
    if anime_name in data['name'].values:
        query_index = data[data['name'] == anime_name].index[0]

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix[query_index], tfidf_matrix)

        # Get recommendations based on similarity scores
        similar_indices = similarity_scores.argsort()[0][-11:-1][::-1]

        recommendations = []
        for index in similar_indices:
            recommended_anime = data.loc[index, 'name']
            recommended_genre = data.loc[index, 'genre']
            recommended_rating = data.loc[index, 'rating']
            recommendation = {
                'anime': recommended_anime,
                'genre': recommended_genre,
                'rating': recommended_rating
            }
            recommendations.append(recommendation)

        return recommendations
    else:
        print(f"The anime '{anime_name}' is not found in the dataset.")


anime_list = pickle.load(open('anime_dict.pkl', 'rb'))
anime = pd.DataFrame(anime_list)


st.title('Anime Recommendation system')
selected_anime = st.selectbox("what do wanna watch", anime['name'].values)

if st.button('Recommend'):
    recommendation = generate_recommendations(selected_anime,data,tfidf_matrix)
    for i in recommendation:
        st.write(i)
