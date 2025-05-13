# Data-Mining-cwk
# Sentiment Analysis Project

This project implements a sentiment analysis pipeline using Python. It processes text data, trains a machine learning model, and predicts whether a given text expresses positive or negative sentiment.

---

## Features

- Text preprocessing (cleaning, tokenization, stopword removal)
- Named Entity Recognition (NER) using spaCy
- Sentiment analysis using TextBlob
- Machine learning models:
  - Logistic Regression (default)
  - Random Forest (optional)
- TF-IDF vectorization for feature extraction
- Model evaluation with confusion matrix and classification report
- Save and load trained models for reuse
- Predict sentiment for new text inputs

---

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.7+
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `nltk`
  - `spacy`
  - `textblob`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `joblib`
  - `tqdm`

Install the required libraries using:

```bash
pip install -r requirements.txt
