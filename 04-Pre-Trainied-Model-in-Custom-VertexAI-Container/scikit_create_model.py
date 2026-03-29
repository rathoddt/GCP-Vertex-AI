# train_sentiment.py
# pip install pandas scikit-learn joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load CSV file (must have "sentiment" and "label" columns)
df = pd.read_csv("gs://sentiment-analysis-training-bucket/input-file/sentiment-analysis-input.csv")

print("Sample data:")
print(df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["sentiment"], df["label"], test_size=0.2, random_state=42
)

# Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save pipeline as joblib (required for Vertex AI sklearn container)
joblib.dump(pipeline, "model.joblib")
print("Saved model.joblib")


# Quick local test
new_text = ["the book is unreliable"]
prediction = pipeline.predict(new_text)
print("\nNew Text:", new_text[0])
print("Predicted Sentiment:", prediction[0])

# sample request in vertex ai console
"""
{
  "instances": [
    "the book is unreliable",
    "I really loved the movie!",
    "The food was terrible"
  ]
}
"""
