##  📈 Financial News Sentiment Analysis

A hybrid sentiment analysis system that predicts the sentiment of financial news headlines using both a Machine Learning classifier (SentenceTransformer + SVC) and Google Gemini generative AI for reasoning. Deployed via Streamlit for easy user interaction.

## 📝 Problem Statement

Financial news affects markets, investor decisions, and trading strategies. Manual sentiment evaluation is slow, subjective, and inconsistent. This project provides a fast, reliable solution by combining:

Machine Learning: Consistent classification of news as Positive, Negative, or Neutral.

Generative AI (Google Gemini): Provides a short explanation of the reasoning behind the sentiment.

Users can input any financial news headline and receive both a prediction and a reasoning explanation instantly.

## 🌟 Features

Text preprocessing and cleaning

Sentence embeddings using all-MiniLM-L6-v2 (SentenceTransformers)

Handling imbalanced classes with SMOTE

Model training: SVC (final saved model) and Logistic Regression

Streamlit web interface for real-time predictions

Gemini API for AI-generated explanations

## 🗂 Project Structure
├── app.py                 # Streamlit app

├── model.py               # ML training pipeline

├── preprocess.py          # Text cleaning & vectorization

├── main.py                # Unit test template

├── sentiment.pkl          # Saved ML model (SVC)

├── transformer.pkl        # Saved sentence transformer

├── sentimentAnalysis.TXT  # Dataset

├── requirements.txt       # Python dependencies

└── README.md              # Project documentation

## ⚙️ Installation

Clone the repository
git clone <repo-url>
cd <project-folder>

## Install dependencies

pip install -r requirements.txt

## Set up Gemini API key

#### Create a .env file with:

GEMINI_API_KEY=your_api_key_here

## ▶ Running the App

Start the Streamlit web app:

streamlit run app.py


Enter a financial news headline in the text area and click Analyze Sentiment. The app will show:

ML Sentiment Prediction (Positive, Negative, Neutral)

Gemini AI Explanation

## 🖥 Example Output

### Input:

“Nvidia reports record-breaking earnings for Q4.”

### ML Prediction:
🟢 Positive

### Gemini Explanation:

Positive — earnings exceeded expectations, indicating strong financial performance.

### 📸 Screenshots / GIF

Streamlit App Home Screen:


Sentiment Prediction Example:


Gemini AI Explanation:


## GIF of App in Action (Optional):


![SharedScreenshot6](https://github.com/user-attachments/assets/6b7b939d-189a-4da9-a548-07105ccbc27b)

Replace the placeholder URLs with your actual screenshots or GIFs from the app.

## 🧠 How It Works
### 1️⃣ Preprocessing (preprocess.py)

Removes non-alphabetic characters

Lowercases and tokenizes text using NLTK

Cleans data for ML model input

### 2️⃣ Model Training (model.py)

Loads dataset (sentimentAnalysis.TXT)

Converts labels to numeric values: Positive=1, Negative=-1, Neutral=0

Embeds sentences with SentenceTransformer

Balances classes using SMOTE

Trains SVC (best-performing model)

Saves model (sentiment.pkl) and transformer (transformer.pkl)

## 3️⃣ Streamlit App (app.py)

Loads ML model and transformer

Cleans and embeds user input

Predicts sentiment via SVC

Sends input to Gemini for AI-generated explanation

Displays results with emojis for visual clarity

## 📦 Requirements
- pandas
- numpy
- nltk
- scikit-learn
- joblib
- streamlit
- sentence_transformers
- imblearn
- imbalanced-learn
- google-generativeai
- dotenv

## ⚠ Limitations

Currently supports English financial news only

Gemini explanations depend on API availability and may vary

Real-world accuracy depends on dataset quality

## 🔮 Future Enhancements

Add RNN/LSTM or transformer-based models for improved accuracy

Support multiple languages

Deploy as a web service (Streamlit Cloud / Docker)

Add historical news sentiment dashboard

## 🙏 Acknowledgments

Sentence Transformers for embeddings

Google Gemini for AI-based explanations

Streamlit for UI

Imbalanced-Learn for SMOTE oversampling

