# 🌊 River Health Monitoring & Pollution Prediction using AI

This project simulates and analyzes river health based on industrial waste impact using synthetic IoT-style environmental data. It focuses on data preprocessing, machine learning modeling using an expanded seven-feature set, and AI-driven pollution flag prediction.

---

## 🚀 Overview

Industrial activities can significantly affect river water quality. This project generates synthetic data mimicking IoT sensor readings from different factory industries, incorporating six critical water quality metrics: *pH*, *Nitrate*, *Water Temperature*, *Turbidity*, *Dissolved Oxygen (DO)*, and *Conductivity*. The core objective is to accurately predict whether a given reading indicates pollution or healthy conditions using advanced classification models.

---

## 🧠 Features

- 🧩 **Synthetic Dataset Generation** using realistic environmental and industrial parameters for 7 features.  
- 🧼 **Data Preprocessing** for handling missing values, scaling, and encoding the expanded feature set.  
- 🤖 **Machine Learning Models** (Random Forest, MLP, LSTM) for high-accuracy **pollution flag prediction**.  
- 📊 **Evaluation Metrics** including Accuracy, Precision, Recall, and F1-Score  
---

## 📂 Repository Structure

```
📁 River-Health-Prediction/
│
├── synthetic_data.py                 # Script for generating synthetic industrial waste data
├── synthetic_river_health_data.csv   # Generated synthetic dataset
├── river_health_preprocessed.csv     # Preprocessed dataset (ready for ML models)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── main.ipynb


---

## 🧰 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/River-Health-Prediction.git
cd River-Health-Prediction
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Generate the synthetic dataset
```bash
python synthetic_data.py
```

This will create a file named `synthetic_river_health_data.csv` in the root folder.

---

## 🧬 Dataset Description

| Column Name           | Description |
|------------------------|-------------|
| **Timestamp**          | Timestamp of the reading (hourly intervals) |
| **Factory_ID**         | Unique ID for each factory |
| **Industry_Type**      | Type of factory (chemical, textile, food_processing) |
| **pH**                 | Water acidity/basicity (range 3–10) |
| **Nitrate_Concentration** | Concentration of nitrates in mg/L (0–100) |
| **Temperature**        | Water temperature in °C (5–40) |
| **Pollution_Flag**     | Target variable (1 = Polluted, 0 = Clean) |


---

## 🧩 Model Workflow

1. **Data Generation** → Using `synthetic_data.py`
2. **Preprocessing** → Handle missing values, normalize, and encode features
3. **Model Training**  
   - **ANN (Artificial Neural Network)** for basic classification  
   - **LSTM (Long Short-Term Memory)** for temporal pattern recognition
4. **Evaluation** → Metrics & visual performance comparison
5. **Deployment (Optional)** → Streamlit app for live predictions

---

## 🧠 Example Use Case

Predict pollution probability based on new sensor readings:
```python
from model import predict_custom

result = predict_custom("chemical", pH=6.2, nitrate=42.5, temperature=30.0)
print(result)
```

Output:
```
Predicted Pollution Flag: 1 (Polluted)
Confidence: 87.4%
```

---

## 📊 Example Visualization

- pH vs Nitrate Concentration
- Pollution Trends across Industries
- Model Performance (Accuracy, F1-score)
- AUC Curves for Classification Models

---

## 🧾 Requirements

- Python 3.9+
- pandas
- numpy
- faker
- scikit-learn
- tensorflow / keras
- matplotlib
- seaborn
- streamlit *(optional)*

---

## 💡 Future Enhancements

- Integration with **real IoT sensor data**
- Predictive maintenance and anomaly detection
- Geospatial analysis of river networks
- Deployment on **AWS / Azure Cloud**

---

## 👨‍💻 Author

**Sai Chetan Reddy**  
🔗 [GitHub](https://github.com/saichetanreddy07)  
💼 Data Science & AI | IoT-Driven Environmental Analytics  

---
