import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)
data = data[["text", "label"]]
data["text"] = data["text"].str.lower()

def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

data["text"] = data["text"].apply(clean_text)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data["text"])
y = data["label"]

X_train, X_text, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved!")


    
