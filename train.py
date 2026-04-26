import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Import CSV file
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

#labelling
fake["label"] = 0
true["label"] = 1

#Merging and Cleaning
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)
data = data[["text", "label"]]
data["text"] = data["text"].str.lower()
data = data.sample(25000)

stop_words = set([
    "the","is","in","on","at","to","a","an","of","and","for",
    "with","by","from","as","that","this","it","was","are",
    "be","has","had","have","but","or","if","they","their",
    "he","she","we","you","i","just","said","would","could",
    "tuesday","monday","wednesday","bbc","news","com","friday","saturday","sunday","thursday"
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 3]
        
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

#Training Model
vectorizer = TfidfVectorizer(
    stop_words = 'english',
    max_df = 0.7,
    ngram_range=(1,2),
    max_features=5000
    )

X = vectorizer.fit_transform(data["text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

#Evaluation
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred))

#Save
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved!")

