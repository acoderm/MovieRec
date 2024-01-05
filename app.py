from flask import Flask, render_template, request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  




@app.route('/get_recommendations', methods=['GET', 'POST'])
def get_recommendations():
    movies = 'movies.csv'
    ratings = 'ratings.csv'

    df_movies = pd.read_csv(movies, usecols=['movieId','title','movie.image_url'], dtype={'movieId':'int32','title':'str','movie.image_url':'str'})
    df_ratings = pd.read_csv(ratings, usecols=['userId','movieId','rating'], dtype={'userId':'int32','movieId':'int32','rating':'float32'})

    fılm_kullanıcılar = df_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    matrix_fılm_kullanıcılar = csr_matrix(fılm_kullanıcılar.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model_knn.fit(matrix_fılm_kullanıcılar)

    def recommender(movie_name, data, model, n_recommendations):
        model.fit(data)
        idx = process.extractOne(movie_name, df_movies['title'])[2]
        distances, indices = model.kneighbors(data[idx], n_neighbors=n_recommendations)
        recommended_movies = [df_movies['title'][i] for i in indices]
        return recommended_movies
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommender(movie_name, matrix_fılm_kullanıcılar, model_knn, 20)
        return render_template('recommendations.html', recommendations=recommendations)
    else:
        return render_template('index.html')  
    
    
  

if __name__ == '__main__':
    app.run(debug=True)
