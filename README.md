# 🛒 Amazon Review Sentiment Analysis using Bi-LSTM

A Deep Learning based NLP project that classifies Amazon product reviews into **Positive**, **Negative**, and **Neutral** sentiments using a Bidirectional LSTM (Bi-LSTM) model.

The project includes data preprocessing, exploratory data analysis (EDA), text tokenization, sequence padding, model training, evaluation, and deployment using Streamlit.

---

## 📌 Project Overview

Understanding customer reviews is important for businesses to analyze customer satisfaction and improve products.

This project predicts the sentiment of Amazon product reviews into three categories:

- 😊 Positive
- 😐 Neutral
- 😔 Negative

The model is built using **TensorFlow/Keras** with a **Bidirectional LSTM** architecture.

---

## 🚀 Features

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Text Preprocessing
- Tokenization
- Sequence Padding
- Bidirectional LSTM Model
- Early Stopping
- Model Evaluation
- Streamlit Web Application
- Saved Model (.keras)
- Saved Tokenizer (.pkl)

---

## 📂 Dataset

The dataset contains Amazon product reviews with corresponding sentiment labels.

### Sentiment Classes

| Label | Sentiment |
|--------|-----------|
| 0 | Positive |
| 1 | Negative |
| 2 | Neutral |

---

## 📊 Exploratory Data Analysis

The following analyses were performed:

- Dataset Overview
- Missing Value Analysis
- Duplicate Value Removal
- Class Distribution
- Review Length Distribution
- Text Statistics
- Sentiment Distribution Visualization

---

## 🧹 Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove unwanted characters
- Clean review text
- Tokenization
- Vocabulary Creation
- Sequence Conversion
- Padding Sequences

---

## 🤖 Model Architecture

```
Input Review
      │
Tokenizer
      │
Text to Sequences
      │
Padding (Max Length = 224)
      │
Embedding Layer
      │
Bidirectional LSTM (128)
      │
Dropout
      │
Bidirectional LSTM (64)
      │
Dropout
      │
Dense Layer
      │
Softmax Output
      │
Prediction
```
## 📥 Download Required Files

Due to GitHub file size limitations, the trained model and full dataset are hosted on Google Drive.

### 📁 Dataset
Download the dataset here:

https://drive.google.com/file/d/1U0yTWsyLHIGudQruQ4_CRqasTm3I0XRy/view?usp=drive_link

### 🤖 Trained Model
Download the trained Bi-LSTM model here:

https://drive.google.com/file/d/16Eb5pRfSTpaB4CKe5D6un-Me__EmCbGh/view?usp=drive_link

After downloading, place the files inside the project folder.

```
amazon-review-sentiment-analysis-bilstm/

├── AmazonReviewPrediction.ipynb
├── app.py
├── amazon_reviews_prediction_model.keras
├── tokenizer.pkl
├── requirements.txt
└── README.md
```
---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📁 Project Structure

```
amazon-review-sentiment-analysis-bilstm/

│
├── AmazonReviewPrediction.ipynb
├── app.py
├── amazon_reviews_prediction_model.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md
└── sample_dataset.csv
```

---

## 📈 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 💻 Streamlit App

The Streamlit application allows users to:

- Enter an Amazon product review
- Predict the review sentiment
- Display prediction confidence
- Show probabilities for all sentiment classes

---

## ▶️ Installation

Clone the repository

```bash
https://github.com/Sahil1576/amazon-review-sentiment-analysis-bilstm
```

Move into the project directory

```bash
cd amazon-review-sentiment-analysis-bilstm
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
streamlit
```

---

## 📌 Future Improvements

- Attention-based Bi-LSTM
- GRU Comparison
- Transformer-based Models (BERT)
- Hyperparameter Optimization
- Model Deployment on Streamlit Cloud

---

## 👨‍💻 Author

**Sahil Katve**

If you found this project useful, consider giving it a ⭐ on GitHub.
