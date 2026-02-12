# Wine Quality Classification using Machine Learning

## Overview
This project focuses on building a machine learning pipeline for predicting wine quality using physicochemical characteristics from the popular Wine Quality dataset.  
Both **red and white wine datasets** were analyzed, preprocessed, and transformed into a binary classification problem to determine whether a wine is **Good** or **Bad** based on quality scores.

The complete workflow — including preprocessing, exploratory data analysis (EDA), model training, and evaluation — was implemented using **Python in Google Colab**.

---

## Key Features
- Data preprocessing and validation pipeline
- Missing value detection and data type verification
- Feature scaling using StandardScaler
- Target variable transformation for binary classification
- Exploratory data analysis and visualization
- Training and evaluation of multiple machine learning models
- Model comparison using performance metrics
- Hyperparameter tuning using GridSearchCV

---

## Technologies Used
- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Google Colab

---

## Data Preprocessing
- Verified missing values and data consistency
- Checked and corrected column data types
- Scaled numerical features using **StandardScaler**
- Converted the quality score into binary classification:
  - Quality ≥ 6 → Good (1)
  - Quality < 6 → Bad (0)

---

## Exploratory Data Analysis
- Correlation matrix heatmap of features
- Wine quality distribution by wine type
- Pairwise feature relationship visualization
- Distribution of wine types (red vs white)
- Alcohol content analysis across quality levels

---

## Machine Learning Models
The following supervised classification models were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- Neural Network (MLPClassifier)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The **Random Forest model** achieved the best performance and was further optimized using **GridSearchCV** for hyperparameter tuning.
