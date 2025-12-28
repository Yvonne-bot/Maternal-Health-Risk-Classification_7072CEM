# Maternal Health Risk Classification Using Machine Learning

## Project Overview

Maternal mortality remains a significant global health challenge, particularly in low- and middle-income countries (LMICs) where early risk detection and access to healthcare resources are limited. This project applies machine learning classification techniques to predict maternal health risk levels (**Low**, **Mid**, **High**) using routinely collected clinical indicators.

The objective is to assess whether data-driven models can support early identification of high-risk pregnancies and contribute to decision-support systems in resource-constrained healthcare environments. This work aligns with the World Health Organization’s Sustainable Development Goal (SDG) 3.1, which focuses on reducing preventable maternal deaths.

This repository contains the implementation and analysis associated with my MSc Machine Learning coursework and accompanying research paper.

---

## Objectives

- Develop and compare multiple machine learning models for maternal health risk classification  
- Address class imbalance to improve minority-class prediction  
- Evaluate model performance using accuracy, precision, recall, F1-score, and AUC  
- Compare individual models with ensemble learning approaches  
- Assess model robustness and suitability for low-resource healthcare contexts  

---

## Dataset

- **Source:** UCI Machine Learning Repository  
- **Number of records:** 1,014 maternal health observations  
- **Predictor features (6):**
  - Age  
  - Systolic Blood Pressure  
  - Diastolic Blood Pressure  
  - Blood Sugar (mmol/L)  
  - Body Temperature  
  - Heart Rate  
- **Target variable:** Maternal health risk level (Low, Mid, High)

### Data Characteristics and Limitations

- No missing values  
- High proportion of duplicate records (approximately 55%)  
- Mild class imbalance, with fewer high-risk cases  
- Limited contextual features such as socio-economic and environmental factors  

---

## Methodology

### Data Preprocessing

- Categorical target labels encoded numerically  
- Feature scaling applied using StandardScaler to normalise variable ranges  
- Outlier analysis conducted using boxplots  
- Correlation analysis performed; highly correlated blood pressure features were retained due to the small dataset size  

### Handling Class Imbalance

- Synthetic Minority Over-sampling Technique (SMOTE) used to balance class distributions and improve minority-class learning  

### Machine Learning Models

The following classification models were implemented and evaluated:

- Logistic Regression  
- Naive Bayes  
- K-Nearest Neighbours (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- XGBoost  

### Ensemble Learning

- Voting Classifier using soft voting  
- Stacking Classifier with a Random Forest meta-model  

### Model Optimisation and Validation

- Hyperparameter tuning using grid and random search strategies  
- Stratified 10-fold cross-validation applied to ensure robust evaluation on imbalanced data  

---

## Results and Findings

| Model | Average 10-Fold Accuracy | AUC |
|------|--------------------------|-----|
| Random Forest | 86.68% | 0.93 |
| XGBoost | 86.17% | 0.94 |
| Voting Classifier | 85.45% | 0.94 |
| Stacking Classifier | 84.94% | 0.95 |

### Key Observations

- Tree-based models (Random Forest and XGBoost) consistently outperformed linear and distance-based models  
- Ensemble methods provided marginal performance improvements but showed signs of overfitting  
- High-risk cases achieved stronger recall than precision, indicating the need for further refinement  
- Random Forest offered the most balanced trade-off between accuracy, robustness, and interpretability  

---

## Ethical Considerations

- The dataset is fully anonymised and contains no personally identifiable information  
- Models are designed as decision-support tools and are not intended to replace clinical judgement  
- Bias and class imbalance were addressed using SMOTE and stratified cross-validation  
- Feature importance analysis was used to improve model transparency  

---

## Future Work

- Incorporation of socio-economic and environmental factors to improve predictive performance  
- Application of model explainability techniques such as SHAP or LIME  
- Evaluation of additional algorithms including LightGBM, CatBoost, and neural networks  
- Validation on external or real-world clinical datasets  

---

## Tools and Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## Repository Structure

```
Maternal-Health-Risk-Classification
│── MATERNAL_HEALTH_RISK_CLASSIFICATION_ML.ipynb
│── Dataset (UCI Maternal Health Risk Data)
│── Research Paper (PDF)
│── README.md
```

---

## Author

**Yvonne Musinguzi**  
MSc Data Science and Machine Learning  
Coventry University  
