from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load trained Naïve Bayes model and TF-IDF vectorizer
with open("model/model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/vectorizer1.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = word_tokenize(text)  # Tokenize words
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(filtered_words)

# Define route for homepage
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        news_text = request.form["news_text"]  # Get user input
        processed_text = preprocess_text(news_text)  # Preprocess input
        text_tfidf = vectorizer.transform([processed_text])  # Convert to TF-IDF
        prediction = model.predict(text_tfidf)[0]  # Predict
        
        result = "Fake News" if prediction == 1 else "Real News"
        return render_template("index.html", result=result, news_text=news_text)

    return render_template("index.html", result=None)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)