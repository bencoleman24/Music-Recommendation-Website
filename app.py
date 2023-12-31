# Import Statements
import csv
from datetime import timedelta
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, session
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import logging

# This function helps get the spotify api credentials by grabbing them from the txt file
def load_env_vars(filename):
    with open(filename, 'r') as f:
        for line in f:
            name, value = line.strip().split('=', 1)
            os.environ[name] = value
            
load_env_vars("env.txt")

# Variables used to access spotify api, will be globally accessed in functions when needed
CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')

# Dataset used for ML reccomendation models
spotify_data = pd.read_csv("data/main_dataset.csv")
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'hip hop', 'jazz', 'classical', 'metal', 'reggae','instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'popularity']
spotify_data[audio_features] = spotify_data[audio_features].fillna(0)

# This function searches the spotify api for a song and returns a dataframe of the songs relevant data
def search_spotify(artist_name, song_title):
    print("Looking up song on Spotify...")

    global CLIENT_ID
    global CLIENT_SECRET

    credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)

    query = 'track: {} artist: {}'.format(song_title, artist_name)
    search_results = spotify.search(q=query, limit=1)
    if search_results['tracks']['items']:
        raw_song_data = search_results['tracks']['items'][0]
        features = spotify.audio_features(raw_song_data['id'])[0]
        
        artist_id = raw_song_data['artists'][0]['id']
        artist_genres = spotify.artist(artist_id)['genres']
        
        song_info = {
            'name': song_title,
            'explicit': int(raw_song_data['explicit']),
            'duration_ms': raw_song_data['duration_ms'],
            'popularity': raw_song_data['popularity'],
            'genres': artist_genres
        }
        song_info.update(features)

        genres = ['pop', 'country', 'dance', 'electronic', 'folk', 'rock',
                'hip hop', 'jazz', 'classical', 'metal', 'reggae']

        for genre in genres:
            song_info[genre] = 0

        song_info_df = pd.DataFrame([song_info])
        
        for genre in genres:
            if genre in artist_genres:
                song_info_df[genre] = 1

        return song_info_df
    else:
        print("No results found for that artist and song title.")
        return pd.DataFrame()


# The purpose of this function is to get the row of data associated with the artist/track 
def get_song(artist, track_name):
    # reformat artist so it fits the style in the dataset
    artist_formatted = f"['{artist}']"
    songs_matching = spotify_data[(spotify_data['name'] == track_name) & (spotify_data['artists_names'] == artist_formatted)]
    if songs_matching.size == 0:
        print("song not found in data")
        try:
            song_data = search_spotify(artist, track_name)
        except:
            return "Error occurred searching spotify api"
        return song_data
    return pd.DataFrame(songs_matching.iloc[0]).T.sort_values(by='release_date', ascending=False)

# This function is used to grab for songs from a playlist in spotify
def get_playlist_tracks(playlist_url):
    playlist_id = playlist_url.split('/')[-1]

    global CLIENT_ID
    global CLIENT_SECRET
    
    credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)
    
    results = spotify.playlist_tracks(playlist_id)
    
    songs = []
    for track in results['items']:
        song_data = track['track']
        
        song_info = {
            'name': song_data['name'],
            'explicit': int(song_data['explicit']),
            'duration_ms': song_data['duration_ms'],
            'popularity': song_data['popularity'],
            'artists': song_data['artists'][0]['name'],
        }

        songs.append(song_info)
    return pd.DataFrame(songs)

# This function get's the links from spotify of the recommendations previously generated
def get_links(recommendations):
    link_list = []
    
    global CLIENT_ID
    global CLIENT_SECRET
    credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)


    for song in recommendations:
        parts = song.strip("'").split(':')

        if len(parts) == 2:
            artist = parts[0].strip()
            track_name = parts[1].strip()
        else:
            print("Invalid input format")
        query = 'track: {} artist: {}'.format(track_name, artist)
        data = spotify.search(q=query, limit=1)['tracks']['items'][0]
        id = data["id"]
        link = f"https://open.spotify.com/track/{id}"
        link_list.append(link)

    
    return link_list 

# This function gets popularity and genre based recommendations
def popularity_genre_recommendation(genre, num_recommendations):
    if genre == "Pop":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M'
    if genre == "Country":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1DX1lVhptIYRda'
    if genre == "Rock":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1EQpj7X7UK8OOF'
    if genre == "Hip-Hop/Rap":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1EIgbjUtLiWmHt'
    if genre == "Electronic/Dance":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1EQp9BVPsNVof1'
    if genre == "R&B":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1EQoqCH7BwIYb7'
    if genre == "Jazz":
        playlist_link = 'https://open.spotify.com/playlist/37i9dQZF1EQqA6klNdJvwx'
    if genre == "Classical":
        playlist_link = 'https://open.spotify.com/playlist/1h0CEZCm6IbFTbxThn6Xcs'

    playlist_df = get_playlist_tracks(playlist_link)
    playlist_df_shuffled = playlist_df.sample(frac=1)
    recommendations = playlist_df_shuffled[['name', 'artists']].values.tolist()[:num_recommendations]
    
    return ["{}: {}".format(artist, song) for song, artist in recommendations]

# This function gets recommendations based on the cosine similarity of audio features
def cosine_similarity_model(artist, track_name, num_recommendations, spotify_data):
    global audio_features 

    song_data = get_song(artist, track_name)
    if song_data is str:  
        return song_data

    song_data[audio_features] = song_data[audio_features].fillna(0)

    cosine_similarities = cosine_similarity(song_data[audio_features].values.reshape(1, -1), spotify_data[audio_features])
    
    # Get the top n most similar tracks (excluding the chosen track itself)
    similar_tracks = spotify_data.iloc[cosine_similarities[0].argsort()][['name', 'artists_names']].drop_duplicates()
    similar_tracks['artists_names'] = similar_tracks['artists_names'].apply(lambda x: x.strip("[]").replace("'", ""))
    similar_tracks_list = similar_tracks[~((similar_tracks['name'] == track_name))].values.tolist()
    recommendations = similar_tracks_list[-num_recommendations:]
    
    return ["{}: {}".format(artist, song) for song, artist in recommendations]

# This function gets recommendations based on the kmeans clustering model
def kmeans_recommendation(artist, track_name, num_recommendations, spotify_data):
    global audio_features 

    song_data = get_song(artist, track_name)
    
    if not isinstance(song_data, pd.DataFrame):
        raise ValueError("Song data retrieval failed.")

    song_data = song_data[audio_features].fillna(0)
    common_columns = spotify_data.columns.intersection(song_data.columns)
    combined_data = pd.concat([spotify_data, song_data[common_columns]], ignore_index=True)

    # Scale data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data[audio_features])
    spotify_scaled = combined_data_scaled[:-1]
    song_data_scaled = combined_data_scaled[-1].reshape(1, -1)

    # KMeans clustering
    kmeans = KMeans(n_clusters=10)  # or determine the optimal number of clusters here
    spotify_data['cluster'] = kmeans.fit_predict(spotify_scaled)

    cluster = kmeans.predict(song_data_scaled)[0]

    same_cluster_tracks = spotify_data[spotify_data['cluster'] == cluster].sample(num_recommendations)
    same_cluster_tracks['artists_names'] = same_cluster_tracks['artists_names'].apply(lambda x: x.strip("[]").replace("'", ""))
    recommendations = same_cluster_tracks[['name', 'artists_names']].values.tolist()
    return ["{}: {}".format(artist, song) for song, artist in recommendations]

# This function gets recommendations based on a nueral network model
def nn_recommendation(artist, track_name, num_recommendations, spotify_data):
    global audio_features 

    input_song_data = get_song(artist, track_name)
    
    if not isinstance(input_song_data, pd.DataFrame):
        return "Error: Song data retrieval failed."

    input_features = input_song_data[audio_features].fillna(0)

    # Standardize the features
    scaler = StandardScaler()
    input_features_scaled = scaler.fit_transform(input_features)

    #Create a neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(len(audio_features),)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(audio_features), activation='linear'))  # Output layer

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    #Train the model on the existing dataset
    X_train = spotify_data[audio_features].fillna(0)
    X_train_scaled = scaler.fit_transform(X_train)

    model.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32)

    # Use the trained model to find similar songs
    input_features_scaled = input_features_scaled.reshape(1, -1)
    predicted_features_scaled = model.predict(input_features_scaled)
    
    # Find songs with features similar to the predicted features
    cosine_similarities = cosine_similarity(predicted_features_scaled, X_train_scaled)

    # Get the top n most similar tracks (excluding the chosen track itself)
    similar_tracks = spotify_data.iloc[cosine_similarities[0].argsort()][['name', 'artists_names']].drop_duplicates()
    similar_tracks['artists_names'] = similar_tracks['artists_names'].apply(lambda x: x.strip("[]").replace("'", ""))

    similar_tracks_list = similar_tracks[~((similar_tracks['name'] == track_name))].values.tolist()
    recommendations = similar_tracks_list[-num_recommendations:]
    print(recommendations)
    return ["{}: {}".format(artist, song) for song, artist in recommendations]



app = Flask(__name__)
app.secret_key = '1e4839w583eifn5927492ej048b10u' 
#app.config["SERVER_NAME"] = "mlmusicrec.com"
app.config['SESSION_COOKIE_SECURE'] = False
#app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    rate_limit_sec = 20
    current_time = time.time()
    
    if 'last_recommendation_time' in session:
        time_since_last_request = current_time - session['last_recommendation_time']
        
        if time_since_last_request < rate_limit_sec:
            return jsonify({"status": "failure", "message": "Please wait a few seconds before requesting another recommendation."})

    session['last_recommendation_time'] = current_time
    print("Updated session time to:", session['last_recommendation_time'])

    # Rest of your code for handling recommendations

    
    data = {
        "status": "failure",
        "message": "An unknown error occurred."
    }

    try:
        model_type = request.form['modelType']

        if model_type == "popularity_based":
            genre = request.form['genre']
            num_recommendations = int(request.form.get('numRecommendationsPop', 3))
            recommendations = popularity_genre_recommendation(genre, num_recommendations)
            link_list = get_links(recommendations)
            #print(recommendations, link_list)

        elif model_type == "cosine_similarity":
            artist = request.form['artist']
            track_name = request.form['track_name']
            num_recommendations = int(request.form.get('numRecommendationsML', '3'))
            recommendations = cosine_similarity_model(artist, track_name, num_recommendations, spotify_data)
            link_list = get_links(recommendations)

        elif model_type == "kmeans":
            artist = request.form['artist']
            track_name = request.form['track_name']
            num_recommendations = int(request.form.get('numRecommendationsML', '3'))
            recommendations = kmeans_recommendation(artist, track_name, num_recommendations, spotify_data)
            link_list = get_links(recommendations)

        elif model_type == "nn":
            artist = request.form['artist']
            track_name = request.form['track_name']
            num_recommendations = int(request.form.get('numRecommendationsML', '3'))
            recommendations = nn_recommendation(artist, track_name, num_recommendations, spotify_data)
            link_list = get_links(recommendations)

        if isinstance(recommendations, list) and len(recommendations) > 0:
            data["links"] = link_list
            data["status"] = "success"
            data["message"] = "Successfully fetched recommendations."
            data["data"] = recommendations
        else:
            data["message"] = "Unable to find recommendations. Please ensure your input is correct and try again."

    except Exception as e:
        data["message"] = str(e)

    response = jsonify(data)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback = request.form['feedback']
        input_data = request.form['input_data']
        
        model = request.form['model']
        output = request.form['output']
        # Writing feedback data to a CSV file
        with open('feedback_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([input_data, model, output, feedback])
        
        return jsonify({"status": "success", "message": "Feedback submitted successfully."})
    except Exception as e:
        return jsonify({"status": "failure", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 8000)
 
