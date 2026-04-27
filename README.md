# River Health Monitoring and Pollution Prediction using AI

## Overview

This project focuses on predicting river pollution levels based on industrial discharge patterns using machine learning. Since real-world multi-parameter river datasets are limited, a synthetic dataset was generated to simulate IoT-based environmental monitoring.

The system uses multiple water quality indicators to classify whether a given observation represents polluted or safe conditions.

---

## Objective

- Simulate realistic environmental data representing industrial impact on rivers  
- Build a machine learning pipeline for pollution detection  
- Compare multiple models to evaluate performance  
- Demonstrate how AI can be applied to environmental monitoring  

---

## Dataset

The dataset represents sensor readings collected at regular intervals from different industrial sources.

**Features:**
- pH  
- Nitrate  
- Water Temperature  
- Turbidity  
- Dissolved Oxygen (DO)  
- Conductivity  
- Industry Type  

**Target:**
- Pollution_Flag (1 = polluted, 0 = clean)

**Note:**  
The dataset is synthetically generated based on realistic environmental ranges and industrial behavior due to the lack of publicly available datasets with similar granularity.

---

## Methodology

1. **Data Generation**  
   Synthetic data was generated using controlled ranges for each feature to reflect realistic environmental conditions.

2. **Data Preprocessing**  
   - Handling missing values  
   - Feature scaling  
   - Encoding categorical variables  

3. **Model Training**  
   The following models were implemented:
   - Random Forest  
   - Artificial Neural Network (ANN)  
   - LSTM (to capture temporal patterns)

4. **Evaluation**  
   Models were evaluated using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  

---

## Results

The models were compared based on classification performance:

- Random Forest provided stable baseline performance  
- ANN captured non-linear feature interactions  
- LSTM showed improved performance by modeling temporal patterns  

(*Add exact metrics here if available*)

---

## Key Observations

- Dissolved oxygen and turbidity strongly influence pollution prediction  
- Industrial category impacts pollution probability  
- Tree-based models perform well on structured environmental data  

---

## Project Structure
