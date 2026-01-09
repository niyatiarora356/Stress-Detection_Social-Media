# Stress Detection from Social Media Text

## A Machine Learning & NLP Project

---

## Overview

In today’s fast-growing world, mental health issues like stress, anxiety and depression are rising globally. Many people express their emotions and struggles openly on social media platforms such as Reddit, Twitter, and Instagram. However, manually identifying stress patterns from millions of posts is practically impossible.

This project presents a stress detection model that analyzes social media posts and detects signs of stress using TF-IDF Vectorizer and a Voting Classifier ensemble model. The system processes social media text data and classifies posts as stressed or non-stressed.

---

## Objective

To build an automated system that:

* Analyzes emotional patterns in social media posts
* Identifies stress-related language
* Classifies posts as stressed or non-stressed
* Supports early stress detection and mental health monitoring

---

## Features

* Stress detection using Natural Language Processing (NLP)
* TF-IDF based text vectorization
* Ensemble machine learning model using Voting Classifier
* Real-time prediction using Streamlit web application

---

## Tech Stack

**Language:** Python

**Libraries & Tools:**

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit

**Models Used:**

* K-Nearest Neighbors
* Support Vector Machine
* Random Forest
* Logistic Regression
* Naïve Bayes
* Decision Tree

---

## Dataset

* Dataset used: dreaddit_StressAnalysis.csv
* Source: Kaggle

---

## Methodology

### 1. Data Collection

The dataset was collected from Kaggle.

### 2. Data Cleaning & Preprocessing

* Separated text features from numerical features
* Removed columns with null values and treated outliers
* Removed irrelevant numerical columns using correlation matrix
* Cleaned text by removing URLs and extra spaces

### 3. Train-Test Split

* Data split into 80% training and 20% testing using train_test_split

### 4. Feature Scaling

* Applied StandardScaler on numerical features

### 5. Text Processing

* Tokenization
* Stop-word removal
* Lemmatization

### 6. TF-IDF Vectorization

* Used TfidfVectorizer to convert text into numerical vectors

### 7. Feature Fusion

* Combined numeric and text features

### 8. Model Training

Built a Voting Classifier using:

* KNN
* SVM
* Random Forest
* Logistic Regression
* Naïve Bayes
* Decision Tree

Used soft voting to average probabilities for better results.

### 9. Model Evaluation

* Confusion Matrix
* Accuracy Score
* F1 Score

### 10. Deployment

* Integrated trained model and TF-IDF vectorizer into a Streamlit web app
* User enters a social media post → model predicts stress level

---

## Results

The evaluation metrics are as follows:

* Accuracy: 72.7%
* F1 Score: 74.8%

---

## Future Scope

* Use data from multiple sources to improve generalization
* Incorporate multilingual datasets for regional language stress detection
* Develop a complete real-time stress tracking system
* Integrate with college counselling portals or self-help mobile applications

---

## How to Run the Project

```bash
git clone https://github.com/your-username/stress-detection-social-media.git
cd stress-detection-social-media
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

Niyati Arora  
Second Year B.Tech Student  
School of Computer Engineering, KIIT DU  
Bhubaneswar, Odisha  
24051718@kiit.ac.in  

---

## Acknowledgements

* Kaggle Dataset: dreaddit_StressAnalysis.csv

---
