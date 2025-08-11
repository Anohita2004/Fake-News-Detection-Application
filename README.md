# Fake News Classifier using Voting Ensemble

##  Overview
This project is an **AI/ML-based text classification system** that detects whether a given news article is *fake* or *real*.  
It leverages **ensemble learning** to combine predictions from multiple machine learning models, improving overall accuracy and robustness.

The goal is to provide a **reliable, scalable, and interpretable** solution for combating misinformation.

---

## Problem Statement
The internet has made it easy to spread information, but also **misinformation**.  
Fake news can have serious social, political, and economic consequences.  
A robust detection system can help social media platforms, fact-checking agencies, and news portals filter out unreliable content before it spreads.

---

## 🛠️ Approach
### 1. **Data Collection**
- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Features: News title, text body, label (`FAKE` / `REAL`)

### 2. **Preprocessing**
- Text cleaning (lowercasing, punctuation & stopword removal, lemmatization)
- Tokenization
- TF-IDF vectorization for ML models  
  *(If using LSTM, word embeddings like Word2Vec/Glove were generated)*

### 3. **Model Building**
- **Base Models**:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Random Forest
  - *(Optional)* LSTM for sequence learning

- **Ensemble Technique**:  
  - **Voting Classifier** (Hard/Soft voting)
  - Aggregates predictions from all base models to make the final decision.

### 4. **Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix for error analysis

---

## Results
| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 94%      |
| Naive Bayes          | 92%      |
| Random Forest        | 93%      |
| **Voting Ensemble**  | **96%**  |

The ensemble approach outperformed individual models, confirming that **model diversity** enhances predictive performance.

---

##  Tech Stack
- **Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn
- **ML Techniques**: Ensemble Learning, TF-IDF Vectorization
- **Optional Deep Learning**: TensorFlow/Keras (for LSTM variant)

---

##  How to Run
```bash
# Clone the repo
git clone https://github.com/username/fake-news-classifier.git
cd fake-news-classifier

# Install dependencies
pip install -r requirements.txt

# Run the classifier
python fake_news_classifier.py
```

---

##  Key Interview Talking Points
- **Why Ensemble?** → Reduces variance & bias by combining diverse models, improving generalization.
- **Why Voting Classifier?** → Simple yet effective; aggregates strengths of different algorithms.
- **Challenges Faced**:
  - Handling imbalanced data
  - Avoiding overfitting with textual features
- **Potential Improvements**:
  - Incorporate deep learning models like BERT
  - Deploy as an API for real-time classification

---

##  Project Structure
```
📁 fake-news-classifier
 ├── data/                # Dataset
 ├── notebooks/           # Jupyter notebooks for EDA & training
 ├── fake_news_classifier.py
 ├── requirements.txt
 └── README.md
