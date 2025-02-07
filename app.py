import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("log-Reg-10.pkl")  # Change if needed
vectorizer = joblib.load("vectorizer.pkl")  # Change if needed


# Function to preprocess tweets
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\@w+|\#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Function to predict if a tweet is about a disaster
def predict_tweet(tweet):
    cleaned_tweet = preprocess_text(tweet)
    tweet_vector = vectorizer.transform([cleaned_tweet])  # Convert to TF-IDF
    prediction = model.predict(tweet_vector)[0]
    return "🚨 Disaster-related Tweet!" if prediction == 1 else "✅ Not a Disaster-related Tweet."

# Streamlit UI
st.title("🚀 Disaster Tweet Classifier")
st.markdown("Enter a tweet below to check if it's related to a disaster.")

# User input field
user_tweet = st.text_area("Enter Tweet", "")

if st.button("Predict"):
    if user_tweet.strip():
        result = predict_tweet(user_tweet)
        st.success(f"**Prediction:** {result}")
    else:
        st.warning("Please enter a tweet before predicting.")
