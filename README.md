# 🧠 Autism Spectrum Disorder (ASD) Prediction Using Machine Learning

This project aims to predict the likelihood of Autism Spectrum Disorder (ASD) in individuals using machine learning techniques. It leverages key screening features and behavioral traits to develop a predictive model, offering a simple GUI-based interface for practical usage.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Machine Learning Models](#machine-learning-models)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Graphical User Interface](#graphical-user-interface)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## 📊 Project Overview

Autism Spectrum Disorder is a neurodevelopmental condition characterized by challenges with social interaction, communication, and restricted/repetitive behaviors. Early diagnosis can significantly improve outcomes through early intervention. This project applies multiple machine learning models to screen individuals based on their responses to behavioral and medical questions.

---

## 🌟 Features
- Data cleaning and preprocessing pipeline
- Feature selection and dimensionality reduction (PCA)
- Comparison of multiple ML models
- Evaluation with metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Interactive GUI built with Tkinter for real-time ASD prediction

---

## 📁 Dataset

- **Source**: [Autism Screening Adult Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)
- **Attributes**: 
  - Behavioral characteristics (A1–A10)
  - Age, Gender, Ethnicity, Family History, etc.
  - Class label (ASD/Not ASD)

---

## 🛠️ Technologies Used

- **Languages**: Python
- **Libraries**:
  - Pandas, NumPy
  - Scikit-learn
  - Matplotlib, Seaborn
  - Tkinter (GUI)
- **Tools**: Jupyter Notebook / VS Code

---

## 🤖 Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)

Each model was tuned and evaluated to select the best-performing approach.

---

## 🔄 Data Preprocessing

- Handling missing values
- Label Encoding for categorical features
- Feature Scaling (Standardization)
- Feature Selection using correlation and statistical methods
- Dimensionality reduction using PCA

---

## 📈 Evaluation Metrics

Models were evaluated on a test set using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Curve

---

## 🖥️ Graphical User Interface

- Built using **Tkinter**
- User-friendly form with dropdowns and input fields
- Predict button to get ASD result instantly
- Displays prediction result with probability

---

## ⚙️ Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/asd-prediction-ml.git
   cd asd-prediction-ml
