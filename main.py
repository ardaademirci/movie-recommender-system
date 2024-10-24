import pandas as pd
import numpy as np
import requests
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

#   Downloading the Dataset & loading data into DataFrames
url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
response = requests.get(url, stream=True)

#Write the zip file to disk
with open ('ml-latest-small.zip', 'wb') as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

#Extract the csv files
with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
    zip_ref.extractall()

#Load csv to dataframe
ratings = pd.read_csv('ml-latest-small/ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies = pd.read_csv('ml-latest-small/movies.csv', usecols=['movieId', 'title'])


#   Preprocessing the data
data = pd.merge(ratings, movies, on='movieId')

#Creating a user - item matrix
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)

#   Calculating Cos Similarity
user_similarity = cosine_similarity(user_item_matrix)
print(user_similarity)

#   Finding similar users
def find_similar_users(user_id, user_similarity, top_n=5):
    user_index = user_item_matrix.index.get_loc(user_id)
    similar_users = user_similarity[user_index]
    similar_users_indices = np.argsort(similar_users)[::-1][1:top_n+1]
    return user_item_matrix.index[similar_users_indices]

#Find the most similart users for 4
similar_users = find_similar_users(4,user_similarity)
print(f'Similar users for user ID 4: {similar_users}')

#   Generating Movie Recommendations
def generate_movie_recs(user_id, user_similarity, user_item_matrix, top_n=5):
    similar_users = find_similar_users(user_id, user_similarity)
    similar_users_ratings = user_item_matrix.loc[similar_users]
    average_ratings = similar_users_ratings.mean()
    recommended_movies = average_ratings.sort_values(ascending=False).head(top_n)
    return recommended_movies

#Generating recommendations for user ID 4
recommendations = generate_movie_recs(4, user_similarity, user_item_matrix)
print(f'Recommended Movies for user ID 4: {recommendations}')

# Perfomence Evalutaion of the Model
def model_evalutaion(user_id, user_similarity, user_item_matrix, actual_ratings, top_n=5):
    recommendations = generate_movie_recs(user_id, user_similarity, user_item_matrix, top_n)
    common_movies = recommendations.index.intersection(actual_ratings.index)
    precision = len(common_movies) / top_n
    recall = len(common_movies) / len(actual_ratings[actual_ratings>0]) #   Filtering rated movies above 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

#Evaluating the model for user ID 4
actual_ratings = user_item_matrix.loc[4]
precision, recall, f1_score = model_evalutaion(4, user_similarity, user_item_matrix, actual_ratings)

print(f'Precision for User ID 4: {precision}')
print(f'Recall for User ID 4: {recall}')
print(f'F1-Score for User ID 4: {f1_score}')

#   The results show that for user id 4, precision=1.0 means all the predictions are relatable movies to the user.
#   However, Recall and F1-Score being low shows that the model missed many relatable movies that the user has rated.
#   This highlights the trade-off between Precision and Recall&F1-Score