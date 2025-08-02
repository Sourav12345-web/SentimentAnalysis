import gradio as gr
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_and_stem(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def predict_sentiment(review):
    clean_review = clean_and_stem(review)
    vectorized = vectorizer.transform([clean_review])
    prediction = model.predict(vectorized)[0]
    return f"Sentiment: {prediction.capitalize()}"

iface = gr.Interface(fn=predict_sentiment,
                     inputs=gr.Textbox(lines=5, placeholder="Enter your review here..."),
                     outputs="text",
                     title="📝 Sentiment Analyzer",
                     description="Enter a product review to get its sentiment.")
iface.launch()
