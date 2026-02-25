# 🎓 Age & Gender Prediction — IIT Madras Capstone Project

> Capstone project for the **Post Graduate Certificate in Advanced Machine Learning & Cloud** — Indian Institute of Technology, Madras.

---

## 📌 Problem Statement

Predict a person's **age** (regression) and **gender** (classification) from facial image data. The project covers the full ML lifecycle — from exploratory analysis and feature engineering to model building, evaluation, and Flask-based deployment.

---

## 🔍 What's Inside

| Notebook / Folder | Description |
|---|---|
| `EDA_and_Advanced_Visualization.ipynb` | Exploratory data analysis, distribution plots, correlation analysis |
| `Feature_Engg_and_Data_Preparation.ipynb` | Feature engineering, preprocessing, train-test split |
| `S1_Age_Prediction_Regression_model.ipynb` | Sprint 1 — Baseline regression models for age prediction |
| `S1_Gender_Prediction_Classification_model.ipynb` | Sprint 1 — Baseline classification models for gender prediction |
| `S2_Age_Prediction_Regression_model.ipynb` | Sprint 2 — Improved regression with tuning & advanced models |
| `S2_Gender_Prediction_Classification_model.ipynb` | Sprint 2 — Improved classification with tuning & advanced models |
| `Age_Prediction_Flask/` | Flask web app for age prediction inference |
| `Gender_Prediction_Flask/` | Flask web app for gender prediction inference |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-Web%20API-black?logo=flask)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)

- **ML Models**: Regression (Age) — Linear Regression, Random Forest, Gradient Boosting | Classification (Gender) — Logistic Regression, Random Forest, XGBoost
- **Deployment**: Flask REST API for both models
- **Evaluation**: RMSE, MAE (Age) | Accuracy, F1-Score, ROC-AUC (Gender)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/krishnaramadas/IITM_Capstone_Project.git
cd IITM_Capstone_Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app (Age Prediction)
```bash
cd Age_Prediction_Flask
python app.py
```

### 4. Run the Flask app (Gender Prediction)
```bash
cd Gender_Prediction_Flask
python app.py
```

---

## 📊 Project Approach

```
Raw Data
   ↓
EDA & Visualization (distributions, outliers, correlations)
   ↓
Feature Engineering & Preprocessing
   ↓
Sprint 1: Baseline Models (Age Regression + Gender Classification)
   ↓
Sprint 2: Hyperparameter Tuning + Advanced Models
   ↓
Flask Deployment (REST API for inference)
```

---

## 👥 Contributors

- **Krishna Ramadas** — [@krishnaramadas](https://github.com/krishnaramadas)
- **Haridhranath Reddy** — [@harindranathreddy](https://github.com/harindranathreddy)

---

## 🏫 Institution

**Indian Institute of Technology, Madras**
Post Graduate Certificate — Advanced Machine Learning & Cloud

---

## 📄 License

This project was developed for academic purposes as part of the IIT Madras PG programme.

