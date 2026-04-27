# 🚀 VeriNews AI — Explainable Fake News Detection System

## 📌 Overview

**VeriNews AI** is an end-to-end Machine Learning application that detects whether a news article is **REAL or FAKE**, while also providing **human-understandable explanations** behind the prediction.

Unlike basic classifiers, this system focuses on **Explainable AI**, allowing users to see *why* a piece of news is flagged as misleading.

---

## 🔥 Key Features

- 🧠 **ML Model** — TF-IDF + Logistic Regression
- 🌐 **Dual Input Support** — Paste raw text or news article URL
- 📊 **Confidence Score Visualization**
- 🔍 **Explainable AI** — Top contributing words influencing prediction
- 🧾 **Automatic Summarization** of article
- 🎭 **Tone Detection** — Neutral vs Sensational content
- ⚡ **Interactive Flask UI**
- 🧩 **Feature Importance Extraction (Model-based interpretability)**
- 📝 **Logging system** to track predictions

---

## 🧠 Model Details

- **Algorithm:** Logistic Regression  
- **Vectorization:** TF-IDF (Unigrams + Bigrams)  
- **Dataset:** Fake.csv + True.csv  
- **Training Size:** ~25,000 samples  
- **Accuracy:** *Add your actual accuracy here*  

---

## 📸 Screenshots

### 🏠 Home Interface
![Home](screenshots/.png)

### 🟢 Real News Detection
![Real](screenshots/real_result.png)

### 🔴 Fake News Detection
![Fake](screenshots/fake_result.png)

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/ESSA-ANSARI/fake-news-detector.git
cd fake-news-detector

pip install -r requirements.txt
python app.py
Then open in browser:

http://127.0.0.1:5000

## 🧪 How It Works

1. User inputs text or URL
2. Article content is extracted (if URL)
3. Text is cleaned and vectorized using TF-IDF
4. Logistic Regression predicts REAL / FAKE
5. Model coefficients identify important words
6. System generates:
 - Confidence score
 - Key influencing words
 - Summary
 - Tone analysis

## 🚀 Future Improvements

This project is actively being improved. Planned upgrades include:

🤖 Advanced models (BERT / Transformers)
🌍 Real-time news verification using APIs
☁️ Cloud deployment (Render / AWS)
📈 Improved explainability using SHAP/LIME
🧠 Bias detection and credibility scoring
📱 Mobile-friendly UI
🔐 User authentication + saved history

## ⚠️ Disclaimer

This system is intended for educational and experimental purposes only.
Predictions may not always be accurate and should not be used as a sole source for verifying news authenticity.

## 👤 Author

Essa Ansari
Aspiring Data Scientist | Al/ML Enthusiast
Focused on building real-world, impactful ML systems

## ⭐ Support
If you found this project useful, consider giving it a ⭐ on GitHub!
