import streamlit as st
import pickle
from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

df = pd.read_csv('internship_data.csv')

df['TextFeatures'] = df['Title'] + ' ' + df['Company Name'] + ' ' + df['Stipend'].astype(str) + ' ' + df['Location']

def train_model(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['TextFeatures'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf_vectorizer, cosine_sim

def save_model(model_filename, model):
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved as {model_filename}")

def load_model(model_filename):
    with open(model_filename, 'rb') as model_file:
        return pickle.load(model_file)

def predict_recommendations(internship_title, df, model, top_n=5):
    tfidf_vectorizer, cosine_sim = model
    matching_indices = df.index[df['Title'] == internship_title].tolist()

    if not matching_indices:
        return []  

    idx = matching_indices[0]
    tfidf_matrix_input = tfidf_vectorizer.transform([df['TextFeatures'].iloc[idx]])
    cosine_sim_input = linear_kernel(tfidf_matrix_input, tfidf_vectorizer.transform(df['TextFeatures']))

    sim_scores = list(enumerate(cosine_sim_input[0]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_recommendations = [
        {
            'Title': df['Title'].iloc[i],
            'Company Name': df['Company Name'].iloc[i],
            'Stipend': df['Stipend'].iloc[i],
            'Location': df['Location'].iloc[i]
        }
        for i, score in sim_scores[1:top_n + 1]
    ]

    return top_recommendations


model = train_model(df)
model_filename = 'recommendation_model.pkl'
save_model(model_filename, model)


loaded_model = load_model(model_filename)


st.title('Internship Recommendation System')
st.image('job.jpg')
title = st.text_input('Field of Interest', placeholder='Enter your field of interest')

if st.button('Submit', type='primary'):
    if title:
        predictions = predict_recommendations(title, df, loaded_model)
        if predictions:
            st.write(f"Top {len(predictions)} predicted internships for {title}:")
            for prediction in predictions:
                st.write(f"Title: {prediction['Title']}, Company: {prediction['Company Name']}, Stipend: {prediction['Stipend']}, Location: {prediction['Location']}")
        else:
            st.write(f"No match found for {title}")
    else:
        st.write("Please enter a field of interest.")
