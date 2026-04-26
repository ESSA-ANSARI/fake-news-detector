from flask import Flask, request, render_template
import pickle
import numpy as np
from utils import extract_text_from_url


app = Flask(__name__)

with open("model/model.pkl", "rb")as f:
    model = pickle.load(f)
    
with open("model/vectorizer.pkl", "rb")as f:
    vectorizer = pickle.load(f)

def generate_reason(words, label):
    top_words = [w for w, s in words[:5]]

    if label == "FAKE NEWS":
        return f"The model identified linguistic patterns often associated with misleading content, including terms like: {', '.join(top_words)}."
    else:
        return f"The article demonstrates structured and factual reporting patterns, with key indicators such as: {', '.join(top_words)}."

def detect_tone(text):
    emotional_words = ["shocking", "breaking", "exclusive", "alert", "urgent", "unbelievable", "must read" ]
    
    for word in emotional_words:
        if word in text.lower():
            return " Emotional / Sensational"
        
    return "Neutral / Informative"

def summarize_text(text):
    return " ".join(text.split()[:40]) + "..."

def get_top_features(text, vectorizer, model, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    vector = vectorizer.transform([text])
    coef = model.coef_[0]
    
    indices = vector.nonzero()[1]
    scores = []
    
    for i in indices:
        scores.append((feature_names[i], coef[i]))
    
    sorted_words = sorted(scores, key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_words[:top_n]

def predict_proba(texts):
    return model.predict_proba(vectorizer.transform(texts))

def get_source_credibility(url):
    if "bbc" in url or "reuters" in url:
        return "High"
    elif "blog" in url or "wordpress" in url:
        return "Low"
    elif "timesofindia" in url:
        return "Median"
    else:
        return "Unknown"

# ------------------ROUTES-----------------
@app.route('/')
def home():
    return render_template('index.html', prediction_text="" , user_input="")

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['news']
    news = user_input
    source = None
    if user_input.startswith("https"):
        source = get_source_credibility(user_input)
        
    #URL HANDLING
    print("ORIGINAL INPUT:", news)
    
    if user_input.startswith("https"):
        extracted = extract_text_from_url(user_input)
        
        print("EXTRACTED TEXT:", extracted[:300] if extracted else "NONE")
        if extracted:
            news = extracted
        else:
            return render_template('index.html', prediction_text = "⚠️ Could not extract article. Try another URL or paste text manually.", user_input = user_input)
    
    #ML Prediction
    transformed_data = vectorizer.transform([news])
    probability = model.predict_proba(transformed_data)
    confidence = round(max(probability[0]) * 100, 2)
    
    prediction = model.predict(transformed_data)
    
    if prediction[0] == 0:
        label = "FAKE NEWS"
    else:
        label = "REAL NEWS"   
    
    #Explanation
    important_words = get_top_features(news, vectorizer, model)
        
    reason = generate_reason(important_words, label) 
    
    tone = detect_tone(news)
    summary = summarize_text(news)

    fake_words = [word for word, score in important_words if score < -0.3]
    real_words = [word for word, score in important_words if score > 0.3]
       

    result = f"\nVerdict: {label} | Confidence: {confidence:.2f}%"
    #logging
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(news[:200] + " | " + label + "\n")
    
    return render_template(
        'index.html',
        prediction_text=result,
        confidence=confidence,
        user_input=user_input,
        fake_words=fake_words,
        real_words=real_words,
        reason=reason,
        summary = summary,
        tone = tone,
        source = source
        )

if __name__ == "__main__":
    app.run(debug=False)
    
