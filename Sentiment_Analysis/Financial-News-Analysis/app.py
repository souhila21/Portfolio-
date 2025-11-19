import streamlit as st
import pickle
from preprocess import clean_text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import google.generativeai as genai  # Make sure this is imported

# set Gemini API
load_dotenv()
genai.configure(api_key = os.getenv('GEMINI_API_KEY'))

# Load models
with open('sentiment.pkl', 'rb') as f:
    model = pickle.load(f)

with open('transformer.pkl', 'rb') as f:
    embed_model = pickle.load(f)


# Predict the test
def predict(text):
    try:
        processed_text = clean_text(text)
        vectorized_text = embed_model.encode([processed_text])
        prediction = model.predict(vectorized_text)[0]
        sentiment_map = {1: 'Positive', -1: 'Negative', 0: 'Neutre'}
        return sentiment_map.get(prediction, 'Unknown')
    except Exception as e:
        return f" Error in ML Prediction: {e}"
 
def gemini_predict(text):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        Analyze the following financial news and predict the sentiment as Positive, Negative or Neutral 
        Explain the reason in 1-2 lines. 
        
        News: "{text}"
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error in Gemini API: {e}"
 
 # App title
st.title("📈 Financial News Analysis")
st.subheader('Sentiment Analysis using ML and GenAI')

# User Input
user_input = st.text_area("Enter the news:")

# Prediction
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict(user_input)
        sentiment_gemini = gemini_predict(user_input)

        label_map = {"Positive": "🟢 Positive", "Neutral": "🟡 Neutral", "Negative": "🔴 Negative"}
        st.success(f"Predicted Sentiment: {label_map.get(sentiment, sentiment)}")
        
        # Show Gemini explanation
        st.markdown("### 🤖 Gemini AI Analysis")
        st.info(sentiment_gemini)
        
    else:
        st.warning("Please enter a headline to analyze.")
