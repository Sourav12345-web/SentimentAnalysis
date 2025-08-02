<p align="center">
  <a href="https://sourav8595-sentimentanalyser.hf.space" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-brightgreen?style=for-the-badge" alt="Live Demo">
  </a>
</p>

# 📝 Sentiment Analysis using TF-IDF and Naive Bayes

This project performs sentiment analysis on product reviews using Natural Language Processing (NLP) and Machine Learning. It classifies reviews into **Positive**, **Neutral**, or **Negative** sentiments using a Complement Naive Bayes model and TF-IDF vectorization. A simple and interactive **Gradio UI** is provided to test the model.

---

## 🚀 Features

- Cleaned and preprocessed 300,000+ product reviews
- TF-IDF based feature extraction
- Balanced dataset via upsampling
- Model training using GridSearchCV (Complement Naive Bayes)
- Performance comparison with Logistic Regression
- Gradio UI for interactive sentiment predictions
- Live deployment on Hugging Face Spaces

---

## 🗂️ Dataset

The dataset used is `Reviews.csv`, containing Amazon product reviews. Reviews are categorized based on their `Score`:
- 1–2 → Negative  
- 3 → Neutral  
- 4–5 → Positive

---

## 🧠 Models Used

- **Complement Naive Bayes** (with hyperparameter tuning)
- **Logistic Regression** (with class weights)

---

## 📊 Evaluation Metrics

- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix (Visualized using Seaborn)

---

## 🛠️ Tech Stack

- **Python** (Pandas, Scikit-learn, NLTK, Joblib)
- **Gradio** – for UI
- **Matplotlib & Seaborn** – for visualization
- **Hugging Face Spaces** – for deployment

---

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
