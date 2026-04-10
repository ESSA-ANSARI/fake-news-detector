from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    
    transformed_data = vectorizer.transform([news])
    probability = model.predict_proba(transformed_data)
    confidence = max(probability[0]) * 100
    
    prediction = model.predict(transformed_data)
    
    if prediction[0] == 0:
        result = f"FAKE NEWS ({confidence:.2f}% Confident)"
    else:
        result = f"TRUE NEWS ({confidence:.2f}% Confident)"
        
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=False)