# 🧠 VeriNews AI — Explainable Fake News Detection System

🚀 **Live Demo:** https://verinews-ai-v9sh.onrender.com

---

## 📌 Overview

VeriNews AI is a deployed machine learning web application that detects whether a news article is **REAL or FAKE**, while also providing **interpretable explanations** for its predictions.

Unlike black-box models, this system focuses on **transparency and trust**, highlighting the key linguistic patterns influencing each decision.

---

## ⚙️ Key Features

* 🧠 **Machine Learning Model** — Logistic Regression with TF-IDF
* 🌐 **URL Input Support** — Extract and analyze news directly from articles
* 📊 **Confidence Score** — Probability-based prediction output
* 🔍 **Explainable AI (XAI)** — Shows influential words behind predictions
* 🧾 **Auto Summarization** — Generates quick preview of content
* 🎭 **Tone Detection** — Identifies emotional vs neutral writing
* 🏷️ **Source Credibility Indicator** — Basic domain-based trust signal
* ⚡ **Deployed Web App** — Accessible via live public URL

---

## 🧠 Model Details

* **Algorithm:** Logistic Regression
* **Vectorization:** TF-IDF (max_features=3000)
* **Text Processing:** Custom cleaning + stopword removal
* **Dataset:** Combined Fake + Real news dataset
* **Training Size:** ~25,000 samples
* **Performance:** ~97% accuracy *(depends on split)*

---

## 📸 Screenshots

### 🏠 Home Interface

![Home]<img width="1365" height="767" alt="Home" src="https://github.com/user-attachments/assets/44f26962-fb72-42ab-9c3b-cdc225117a01" />

### 🟢 Real News Detection

![Real]<img width="1365" height="767" alt="Real" src="https://github.com/user-attachments/assets/2339b565-9231-43b6-a292-2191cc297023" />

### 🔴 Fake News Detection

![Fake]<img width="1365" height="767" alt="Fake" src="https://github.com/user-attachments/assets/decca40f-ccef-4923-bd3d-f68c32e2d964" />

---

## 🛠️ Tech Stack

* **Backend:** Flask
* **ML Libraries:** scikit-learn, pandas, numpy
* **NLP:** TF-IDF, custom preprocessing
* **Deployment:** Render
* **Other:** newspaper3k, BeautifulSoup

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/ESSA-AI/fake-news-detector.git
cd fake-news-detector

pip install -r requirements.txt
python app.py
```

---

## 🧩 How It Works

1. User inputs text or URL
2. Article content is extracted (if URL)
3. Text is cleaned and preprocessed
4. TF-IDF vectorization applied
5. Logistic Regression predicts label
6. Confidence score generated
7. Top influencing words extracted
8. Summary + tone + explanation displayed

---

## 🚧 Limitations

* Model may struggle with highly nuanced or satirical content
* Source credibility is rule-based (not learned)
* Performance depends on training dataset quality

---

## 🔮 Future Improvements

* 🔬 Integrate SHAP/LIME for advanced explainability
* 🌍 Multi-language support
* 🧠 Transformer-based models (BERT)
* 🎨 Improved UI/UX
* 🔗 Real-time news API integration

---

## ⚠️ Disclaimer

This project is for educational and experimental purposes only.
Predictions may not always be accurate and should not be used as the sole source of truth.

---

## 👤 Author

**Essa Ansari**
Aspiring Data Scientist | AI/ML Enthusiast

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
