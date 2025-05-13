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
```

## Setup

Download required NLP resources:

```bash
# NLTK Downloads
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# spaCy Download
python -m spacy download en_core_web_sm
```

## How to Use

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Prepare the Dataset
Required file format:
- File name: `2.csv` (place in project root)
- Required columns:
  - `Score` (sentiment labels)
  - `Text` (content to analyze)

### 3. Run the Notebook
Execute cells in:  
`text_mining_too (2).ipynb`

### 4. Model Training
Pipeline includes:
1. Text preprocessing
2. Model training (Logistic Regression default)
3. Evaluation:
   - Confusion matrix
   - Classification report

### 5. Save Models
Output files:
- `sentiment_model.pkl` (trained model)
- `tfidf_vectorizer.pkl` (feature extractor)

### 6. Make Predictions
Usage example:
```python
print(predict_sentiment("I love this product!"))  # → "Positive"
print(predict_sentiment("This is terrible"))     # → "Negative"
```

## Optional: Random Forest
To switch classifier:
1. Uncomment Random Forest section
2. Re-run notebook cells

## Outputs
| Component          | Description                          |
|--------------------|--------------------------------------|
| Confusion Matrix   | Visual classification performance   |
| Classification Report | Precision/Recall/F1 metrics      |
| Predictions        | Live text classification            |

## Notes
- Dataset requirements:
  - CSV format
  - UTF-8 encoding
- Preprocessing:
  - Customize `clean_text()` as needed
- Hyperparameters:
  - Adjust `max_iter` (Logistic Regression)
  - Tune `n_estimators` (Random Forest)

## License
MIT License
