# pip install pandas numpy nltk seaborn scikit-learn

import pandas as pd
import numpy as np
import nltk
import re
import seaborn as sns
import sklearn

from nltk.corpus import stopwords
nltk.download('stopwords')  # Ensure stopwords are downloaded
stop_words = set(stopwords.words("english"))

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')  # Ensure it's up to date
nltk.download('punkt_tab')  # Manually download missing punkt_tab

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


file = 'Dataset/top100_kdrama.csv'
try:
    data = pd.read_csv(file)
except FileNotFoundError:
    print(f"Error: The file '{file}' was not found.")
    exit()

# data = data.dropna(subset=["Synopsis", "Genre", "Tags"])  # Handle missing values
data["Rank"] = data["Rank"].str.replace("#", "", regex=True).astype(int)
data["Rating"] = pd.to_numeric(data["Rating"])
data["Genre"] = data["Genre"].str.split(", ")
data["Tags"] = data["Tags"].str.split(", ")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = nltk.word_tokenize(text)
    custom_stopwords = set(stopwords.words("english")).union({
    "drama", "movie", "film", "episode", "series", "kdrama", "korean"})
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in custom_stopwords]
    return " ".join(words)

def recommend_dramas(user_query, num_recommendations=5):
    processed_query = preprocess_text(user_query)

    if not processed_query:  # Handle empty query case
        return "Sorry, I couldn't understand your input. Try using keywords like 'romance' or 'action'."
    
    queryVector = vectorizer.transform([processed_query])

    similarity = cosine_similarity(queryVector, tfidf_matrix).flatten()
    recommended_indices = similarity.argsort()[-num_recommendations:][::-1]
    recommendations = data.iloc[recommended_indices][["Rank", "Name", "Genre", "Rating"]]
    return recommendations, similarity

# gather user input
def user_input():
    query = input("Describe the type of korean drama reccomendation you would like?\n")
    return query

def output(userQuery):
    recommended_dramas, similarity = recommend_dramas(userQuery)

    if isinstance(recommended_dramas, str):  # If error message is returned
        print(recommended_dramas)
    else:
        print("\nTop recommended dramas:")
        for i, row in enumerate(recommended_dramas.itertuples(), start=1):
            similarity_percentage = round(similarity[row.Index] * 100, 2)
            print(f"{i}. {row.Name} {similarity_percentage}%")

data["combined text"] = (
    data["Synopsis"] + " " + 
    data["Genre"].apply(lambda x: " ".join(x)) + " " + 
    data["Tags"].apply(lambda x: " ".join(x) * 2  # Double weight for tags
))

# Preprocess the combined text field in the dataset
data["combined text"] = data["combined text"].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 3))
tfidf_matrix = vectorizer.fit_transform(data["combined text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Run the recommendation system
query = user_input()
output(query)
