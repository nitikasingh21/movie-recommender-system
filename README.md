# Movie Recommender System

Overview
This project implements a movie recommender system using machine learning techniques, specifically cosine similarity for content-based filtering. The system allows users to input their movie preferences and generates personalized recommendations based on the similarity of movie features such as genres, keywords, cast, and crew.

Features
User-friendly Graphical User Interface (GUI) developed with Streamlit for easy interaction.
Data preprocessing to handle missing values, extract relevant features, and create movie tags.
Utilizes CountVectorizer for vectorization of movie tags and cosine similarity for recommendation generation.
Recommendation function to provide top movie recommendations based on user input.
Model persistence using pickle for efficient data storage and retrieval.

Usage
Select a movie title from the dropdown menu in the GUI.
Click on the "Recommend" button to generate personalized movie recommendations.
View the recommended movies displayed in the GUI based on cosine similarity scores.

Technologies Used
Python
Streamlit
pandas
NumPy
scikit-learn
nltk

Future Enhancements
Incorporate collaborative filtering techniques for improved recommendation accuracy.
Enhance user profiling capabilities to capture more personalized preferences.
Integrate external data sources for richer movie metadata and recommendation insights.
