# AI-Driven River Health Monitoring Using Synthetic Industrial Data

## Overview
This project proposes an AI-powered solution for **river health monitoring** using **synthetic datasets** that simulate water pollution from **chemical**, **textile**, and **food processing** industries.  
The goal is to create a realistic dataset to train machine learning models for reliable pollution detection, enabling **real-time**, **data-driven** environmental monitoring.

---

## Key Features
- **Synthetic Data Generation**  
  - Simulates realistic industrial pollution scenarios.
  - Parameters include pH, nitrate concentration, temperature, and dissolved oxygen.
- **Machine Learning Models**  
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Classifier (SVC)
- **Performance**  
  - Logistic Regression achieved the highest accuracy (97%).
- **Exploratory Data Analysis**  
  - Histograms, correlation matrices, outlier detection performed on synthetic data.
- **Streamlit Web App**  
  - Real-time river pollution risk prediction.
  - Risk categories: Very Low, Low, Moderate, High, Very High.

---

## Project Structure
- `/dataset` — Synthetic water quality dataset (CSV)
- `/models` — Pretrained machine learning models (Pickle files)
- `/streamlit_app` — Streamlit frontend app
- `/notebooks` — Jupyter notebooks for data generation, EDA, and model training

---

## Technologies Used
- Python (Pandas, Scikit-learn, Faker, Matplotlib, Seaborn)
- Streamlit (for web app)
- Machine Learning (classification algorithms)
- Data Imputation, Feature Engineering, Synthetic Data Simulation

---

## Future Work
- Integration with **real-time IoT sensor data**.
- Deploying models on **edge devices** for decentralized monitoring.
- Expanding datasets to include parameters like turbidity and biological oxygen demand (BOD).

---
