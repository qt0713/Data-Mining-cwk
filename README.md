# Data-Mining-cwk
Sentiment Analysis Project
This project implements a sentiment analysis pipeline using Python. It processes text data, trains a machine learning model, and predicts whether a given text expresses positive or negative sentiment.

Features
Text preprocessing (cleaning, tokenization, stopword removal)
Named Entity Recognition (NER) using spaCy
Sentiment analysis using TextBlob
Machine learning models:
Logistic Regression (default)
Random Forest (optional)
TF-IDF vectorization for feature extraction
Model evaluation with confusion matrix and classification report
Save and load trained models for reuse
Predict sentiment for new text inputs
Requirements
Before running the project, ensure you have the following installed:

Python 3.7+
Required Python libraries:
numpy
pandas
nltk
spacy
textblob
scikit-learn
matplotlib
seaborn
joblib
tqdm
Install the required libraries using:
pip install -r requirements.txt
Additionally, download the necessary NLTK and spaCy resources:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# For spaCy
python -m spacy download en_core_web_sm
How to Use
1. Clone the Repository
Clone the repository to your local machine:

2. Prepare the Dataset
Place your dataset file (e.g., 2.csv) in the project directory. Ensure the dataset contains at least two columns:

Score: The target column (e.g., sentiment score)
Text: The text data for analysis
3. Run the Notebook
Open the provided Jupyter Notebook (text_mining_too (2).ipynb) in your preferred environment (e.g., Jupyter Notebook, VS Code) and execute the cells step by step.

4. Train the Model
The notebook will:

Preprocess the text data
Train a Logistic Regression model (or Random Forest if enabled)
Evaluate the model using a confusion matrix and classification report
5. Save the Model
The trained model and TF-IDF vectorizer will be saved as:

sentiment_model.pkl
tfidf_vectorizer.pkl
6. Predict Sentiment
Use the predict_sentiment function to classify new text inputs. Example:

Optional: Use Random Forest Classifier
To use the Random Forest Classifier instead of Logistic Regression:

Uncomment the relevant cells in the notebook.
Train and evaluate the Random Forest model.
Outputs
Confusion Matrix: Visualizes the model's performance.
Classification Report: Includes precision, recall, F1-score, and accuracy.
Predictions: Classifies new text as Positive or Negative.
Notes
Ensure the dataset is properly formatted before running the notebook.
Modify the clean_text function if additional preprocessing is required.
Adjust hyperparameters (e.g., max_iter, n_estimators) to optimize model performance.
License
This project is licensed under the MIT License. Feel free to use and modify it as needed.
