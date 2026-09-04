import pandas as pd
import numpy as py
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("shibe123/discord-phishing-scam")

print("Path to dataset files:", path)

#find csv file
file_path = os.path.join(path, "discord-phishing-scam-detection.csv")

#load dataset
df = pd.read_csv(file_path,encoding="latin1")

print("\nDataset loaded")
print("Total messages:", len(df))

#Check what the dataset looks like
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel counts:")
print(df['label'].value_counts())

X = df["msg_content"].astype(str)
y = df["label"]

print("\nMessages:", len(X))
print("Labels:", len(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

#Turn msgs into nums so model understands
vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,2),max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nText converted into numerical features")
print("Training shape:", X_train_tfidf.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(class_weight='balanced')
max_iter=1000
random_state=42

# Train it
model.fit(X_train_tfidf, y_train)

print("\nModel trained successfully")

import joblib

# Save the trained model and vectorizer
joblib.dump(model, "scam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved")

from sklearn.metrics import accuracy_score, classification_report

#make predictions
predictions = model.predict(X_test_tfidf)

#accuracy
accuracy = accuracy_score(y_test, predictions)


print("\nModel accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification report:")
print(classification_report(y_test, predictions))

def check_message(message):
    message_numbers = vectorizer.transform([message])

    prediction = model.predict(message_numbers)[0]
    probability = model.predict_proba(message_numbers)[0][1]

    if prediction == 1:
        print("\nPrediction: Scam/Phishing")
    else:
        print("\nPrediction: Likely Normal")

    print("Scam probability:", round(probability * 100, 2), "%")


while True:
    message = input("\nEnter a Discord message or type quit: ")

    if message.lower() == "quit":
        break

    check_message(message)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Scam"]
)

display.plot()
plt.title("Discord Scam Detection - Confusion Matrix")
plt.show()

import joblib

joblib.dump(model, "scam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved")