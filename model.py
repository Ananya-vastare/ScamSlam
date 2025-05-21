import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset with correct column names
df = pd.read_csv("spam.csv", encoding="latin-1")[["label", "text"]]

# Map 'ham' to 0 and 'spam' to 1
df["label_num"] = df.label.map({"ham": 0, "spam": 1})

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df["text"])

# Model training
model = MultinomialNB()
model.fit(X, df["label_num"])

# Save the trained model and vectorizer
joblib.dump(model, "sms_model.pkl")
joblib.dump(cv, "sms_vectorizer.pkl")

print("Model and vectorizer saved successfully.")
