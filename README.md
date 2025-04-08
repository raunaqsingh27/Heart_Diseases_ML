# Heart Disease Prediction

A machine learning project that analyzes heart disease data to predict the presence of heart disease in patients.

## Project Overview

This project uses machine learning algorithms to predict whether a patient has heart disease based on clinical parameters. The implementation includes data exploration, preprocessing, model training, and evaluation of multiple classification algorithms.

## Dataset

The project uses the Heart Disease dataset (Heart.csv) which contains various features such as:

- Demographic information (age, sex)
- Clinical measurements (blood pressure, cholesterol levels)
- ECG results (resting ECG, max heart rate)
- Chest pain characteristics
- Other medical indicators

## Requirements

To run this project, you need the following libraries:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install these dependencies using:
```
pip install -r requirements.txt
```

## Usage

1. Clone this repository
2. Ensure the Heart.csv file is in the same directory as the script
3. Run the Python script:
```
python Heart_Diseases_Complete_Code.py
```

## Implementation Details

The project implements the following steps:

1. **Data Exploration**
   - Statistical analysis of the dataset
   - Visualization of feature distributions and correlations
   - Analysis of relationships between features and heart disease

2. **Data Preprocessing**
   - Feature scaling using StandardScaler
   - Train-test split (80/20)

3. **Model Training and Evaluation**
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Hyperparameter tuning using GridSearchCV

4. **Model Evaluation Metrics**
   - Accuracy
   - Classification report (precision, recall, f1-score)
   - Confusion matrix
   - ROC curve and AUC

5. **Feature Importance Analysis**
   - Identification of the most predictive features

## Results

The project compares the performance of different machine learning models and identifies the most important features for heart disease prediction. Visualizations include:

- Correlation heatmap
- Feature distribution plots
- Confusion matrices
- ROC curves
- Feature importance chart

## Future Improvements

Potential enhancements for this project:
- Implement cross-validation for more robust evaluation
- Try additional machine learning algorithms
- Perform feature engineering
- Deploy the model as a web application 