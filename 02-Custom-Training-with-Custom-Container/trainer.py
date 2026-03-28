import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("gs://<your-bucket-name>/<your-dataset-full-path-inside-bucket>")   
# columns: sentiment, label

# Map labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["label"].map(label_map)

X = df["sentiment"].values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Text Vectorization Layer
# -----------------------------
max_tokens = 10000
max_len = 100

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_sequence_length=max_len,
    output_mode="int"
)

# Adapt on training text
vectorize_layer.adapt(X_train)

# -----------------------------
# Build Model (with vectorizer inside)
# -----------------------------
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(max_tokens, 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

# -----------------------------
# Save entire pipeline
# -----------------------------
model.save("gs://<your-bucket-name>/<location-inside-bucket>/sentiment_model")  # includes vectorizer + LSTM
