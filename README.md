
# 🏠 House Price Predictor - AI Mini Project

A machine learning-based web app to predict house prices using user-input features like income, house age, and more. The project also compares multiple regression models to select the most accurate one for predictions.

---

## 📌 Overview

This project aims to build a regression-based predictive model using a real-world USA housing dataset. The goal is to compare the performance of various machine learning models and deploy the best one through an interactive user interface.

---

## 🧠 Problem Statement

Predict the house price based on:
- Average Area Income
- Average Area House Age
- Average Area Number of Rooms
- Average Area Number of Bedrooms
- Area Population

---

## 🔍 Model Comparison

We evaluated and compared four regression models:

| Model                | MAE       | RMSE       | R² Score |
|---------------------|-----------|------------|----------|
| Linear Regression    | 80,879.10 | 100,444.06 | **0.9180** ✅ |
| Decision Tree        | 142,942.57| 181,892.04 | 0.7311   |
| Random Forest        | 95,337.40 | 121,001.62 | 0.8810   |
| Gradient Boosting    | 87,404.15 | 109,446.49 | 0.9026   |

➡️ **Linear Regression** was selected as the final model due to its high accuracy (R² = 0.9180) and low error.

---

## 🖥 GUI App

An easy-to-use graphical interface is created using **Streamlit**.

### 🎯 Features:
- User-friendly input form for housing data
- Predict button to estimate house price instantly
- Displays predicted price clearly

---

## 🚀 How to Run

### 🔧 1. Install dependencies

```bash
pip install -r requirements.txt
````

### ▶️ 2. Run the GUI app

```bash
streamlit run gui/app.py
```

---

## 📁 Folder Structure

```
House_Price_Predictor/
├── data/
│   └── USA_Housing.csv
├── models/
│   └── house_price_model.pkl
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── gui/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

* Python
* scikit-learn
* Pandas, NumPy
* Streamlit (for GUI)
* Matplotlib, Seaborn (for EDA)
* Jupyter Notebook

---

## ✨ Highlights

* 📈 Multiple model training & evaluation
* 🥇 Best model selection based on performance
* 🧪 Model saved using `joblib`
* 🖥️ Streamlit GUI to make real-time predictions

---

## 📚 Dataset

* [USA\_Housing.csv](https://www.kaggle.com/datasets/sergeevich/usa-housing)
  *(You can link your own source if dataset differs)*

---

## 👩‍💻 Author

**Karishma Sankar**
📌 AI/ML Enthusiast • Aspiring AI Architect
🔗 [GitHub](https://github.com/karishmasankar) | [LinkedIn](https://www.linkedin.com/in/karishma-sankar1306/)


