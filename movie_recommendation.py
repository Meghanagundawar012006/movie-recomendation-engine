import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies["genre"])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

def recommend(movie_title):
    movie_title = movie_title.lower()

    movie_index = None
    for i, title in enumerate(movies["title"]):
        if title.lower() == movie_title:
            movie_index = i
            break

    if movie_index is None:
        print("Movie not found!")
        return

    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nRecommended movies for '{movies.iloc[movie_index]['title']}':\n")

    count = 0
    for movie in scores[1:]:
        index = movie[0]
        print(movies.iloc[index]["title"])
        count += 1
        if count == 5:
            break

# User Input
movie_name = input("Enter a movie name: ")
recommend(movie_name)