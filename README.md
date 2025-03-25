📰 Fake News Detection Using NLP & Flask
📌 Project Overview
Fake news is a major problem in today's digital world. This project aims to detect fake news articles using Natural Language Processing (NLP) and Machine Learning techniques. We use TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization and Naïve Bayes Classification for prediction. The project is deployed using Flask, making it accessible via a web application.

✨ Features
✅ Classifies news articles as Fake or Real
✅ Uses TF-IDF Vectorization for feature extraction
✅ Built with Naïve Bayes Classifier for efficient classification
✅ Flask-powered web application for user interaction
✅ Supports custom user input (you can paste any news article to check its authenticity)

🚀 Technologies Used
Backend (ML Model & Flask API)
Python: Core programming language

Flask: For deploying the model as a web app

Scikit-Learn: Machine learning library

NLTK (Natural Language Toolkit): For text preprocessing

Pandas & NumPy: Data processing

Frontend (Web UI)
HTML, CSS: For user interface

Bootstrap: For responsive design

JavaScript: To enhance interactivity

📂 Project Structure
graphql
Copy
Edit
📂 Fake-News-Detection
│── 📂 static/                 # Static files (CSS, JS, images)
│── 📂 templates/              # HTML frontend files
│   ├── index.html             # Home page for user input
│   ├── result.html            # Displays the prediction result
│── 📂 model/                  # Contains trained model files
│   ├── model.pkl              # Saved Naïve Bayes classifier
│   ├── vectorizer.pkl         # Saved TF-IDF vectorizer
│── 📂 dataset/                # Raw dataset files
│   ├── Fake.csv               # Fake news dataset
│   ├── True.csv               # Real news dataset
│── app.py                     # Flask application
│── requirements.txt            # Dependencies for installation
│── README.md                   # Project documentation
│── .gitignore                  # Ignores unnecessary files
📊 Dataset Description
We use the Fake.csv and True.csv datasets, which contain real and fake news articles collected from various sources.

Fake.csv → Contains 12,999 fake news articles

True.csv → Contains 21,417 real news articles

Dataset Structure
Column	Description
title	Title of the news article
text	The full news article content
subject	The topic category (e.g., politics, world news)
date	Date of publication
label	1 for Fake News, 0 for Real News (added during preprocessing)
🛠 Installation & Setup
Follow these steps to set up and run the project on your local machine.

1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
2️⃣ Install Dependencies
Ensure you have Python installed, then install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Train the Machine Learning Model
bash
Copy
Edit
python train.py
This will: ✅ Load and preprocess the dataset
✅ Train the Naïve Bayes Classifier
✅ Save the trained model (model.pkl) and vectorizer (vectorizer.pkl)

4️⃣ Run the Flask App
bash
Copy
Edit
python app.py
After running the command, the Flask app will start locally.
🔗 Open in your browser: http://127.0.0.1:5000/

🖥 Usage Guide
1️⃣ Web App Interface
Open the application in a web browser (http://127.0.0.1:5000/).

Enter or paste a news article in the text box.

Click "Check News".

The system will classify it as Real News ✅ or Fake News ❌.

2️⃣ Running Model Directly (Without Flask)
You can test the model from a Python script:

python
Copy
Edit
import pickle

# Load the trained model & vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Function to check news authenticity
def fake_news(news):
    news = vectorizer.transform([news])  # Convert text to TF-IDF
    prediction = model.predict(news)
    return "Fake News" if prediction == 1 else "Real News"

# Example input
news_text = "Government announces new tax policies for 2025."
result = fake_news(news_text)

print("Prediction:", result)


