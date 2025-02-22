# Install necessary libraries before running the script
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
nltk.download('wordnet')  # Ensure lemmatization support
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')  # Ensure tokenization is up to date
nltk.download('punkt_tab')  # Manually download missing punkt_tab (if needed)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from CSV file
def load_data(file):
    try:
        return pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found.")
        exit()

# Function to preprocess text data for NLP tasks
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = nltk.word_tokenize(text)  # Tokenize the text
    
    # Define custom stopwords (removes common, non-informative words)
    custom_stopwords = set(stopwords.words("english")).union({
        "drama", "movie", "film", "episode", "series", "kdrama", "korean"
    })
    
    # Lemmatize words and filter out stopwords and non-alphanumeric tokens
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in custom_stopwords]
    
    return " ".join(words)  # Return cleaned text

# Function to recommend K-dramas based on user input
def recommend_dramas(user_query, data, vectorizer, tfidf_matrix, num_recommendations=5):
    processed_query = preprocess_text(user_query)  # Preprocess user query
    
    if not processed_query:  # Handle empty or uninformative queries
        return "Sorry, I couldn't understand your input. Try using keywords like 'romance' or 'action'."
    
    queryVector = vectorizer.transform([processed_query])  # Transform query into TF-IDF vector
    
    # Compute cosine similarity between user query and dataset
    similarity = cosine_similarity(queryVector, tfidf_matrix).flatten()
    
    # Get top N most similar dramas
    recommended_indices = similarity.argsort()[-num_recommendations:][::-1]
    recommendations = data.iloc[recommended_indices][["Rank", "Name", "Genre", "Rating"]]
    
    return recommendations  # Return recommended dramas

# Function to get user input
def user_input():
    return input("Describe the type of Korean drama recommendation you would like:\n")

# Function to display recommendations
def output(userQuery, data, vectorizer, tfidf_matrix):
    recommended_dramas = recommend_dramas(userQuery, data, vectorizer, tfidf_matrix)
    
    if isinstance(recommended_dramas, str):  # Handle error messages
        print(recommended_dramas)
    else:
        print("\nTop recommended dramas:")
        for i, row in enumerate(recommended_dramas.itertuples(), start=1):
            print(f"{i}. {row.Name}")

# Main function to execute the program
def main():
    file = 'Dataset/top100_kdrama.csv'
    data = load_data(file)
    
    # Convert 'Rank' column from string format (with '#') to integer
    data["Rank"] = data["Rank"].str.replace("#", "", regex=True).astype(int)
    
    # Convert 'Rating' column to numeric format
    data["Rating"] = pd.to_numeric(data["Rating"])
    
    # Convert 'Genre' and 'Tags' to lists
    data["Genre"] = data["Genre"].str.split(", ")
    data["Tags"] = data["Tags"].str.split(", ")
    
    # Combine text fields (Synopsis, Genre, Tags) for TF-IDF processing
    data["combined text"] = (
        data["Synopsis"] + " " +  
        data["Genre"].apply(lambda x: " ".join(x)) + " " +  
        data["Tags"].apply(lambda x: " ".join(x) * 2)  # Double weight for tags
    )
    
    # Apply text preprocessing to the combined text field
    data["combined text"] = data["combined text"].apply(preprocess_text)
    
    # Create TF-IDF vectorizer with n-grams
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 3))
    
    # Convert processed text data into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(data["combined text"])
    
    # Run the recommendation system
    query = user_input()
    output(query, data, vectorizer, tfidf_matrix)

if __name__ == "__main__":
    main()
