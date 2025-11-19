import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')

# Cleaning and tokenization
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Text to Numerical
def text_vector(train_text, test_text):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.fit_transform(test_text)
    return X_train, X_test, vectorizer
