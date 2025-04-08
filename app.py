# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Heart.csv')


# Display first few rows
print(df.head())

# Check dataset information
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Understanding the target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution')
plt.xlabel('0 = No Heart Disease, 1 = Heart Disease')
plt.ylabel('Count')
plt.show()

# Explore correlations between features
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Explore relationships between key features and target
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Age distribution by target
sns.histplot(data=df, x='age', hue='target', bins=20, kde=True, ax=axes[0])
axes[0].set_title('Age Distribution by Target')

# Cholesterol levels by target
sns.boxplot(data=df, x='target', y='chol', ax=axes[1])
axes[1].set_title('Cholesterol Levels by Target')

# Resting blood pressure by target
sns.boxplot(data=df, x='target', y='trestbps', ax=axes[2])
axes[2].set_title('Resting Blood Pressure by Target')

# Max heart rate by target
sns.boxplot(data=df, x='target', y='thalach', ax=axes[3])
axes[3].set_title('Max Heart Rate by Target')

# Gender distribution by target
gender_counts = pd.crosstab(df['sex'], df['target'])
gender_counts.plot(kind='bar', ax=axes[4])
axes[4].set_title('Gender Distribution by Target')
axes[4].set_xlabel('0 = Female, 1 = Male')

# Chest pain type by target
cp_counts = pd.crosstab(df['cp'], df['target'])
cp_counts.plot(kind='bar', ax=axes[5])
axes[5].set_title('Chest Pain Type by Target')

plt.tight_layout()
plt.show()


# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")



# Define a function to evaluate and display model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC (for models that support predict_proba)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        has_roc = True
    except:
        has_roc = False
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve if available
    if has_roc:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()
    
    return accuracy, model

# Create and evaluate Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_accuracy, lr_model = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")

# Create and evaluate Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_accuracy, rf_model = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")

# Create and evaluate SVM model
svm_model = SVC(probability=True, random_state=42)
svm_accuracy, svm_model = evaluate_model(svm_model, X_train_scaled, X_test_scaled, y_train, y_test, "Support Vector Machine")


# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create and fit the grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
print(f"\nBest Random Forest Parameters: {best_params}")

# Evaluate the best model
best_rf_model = grid_search.best_estimator_
best_rf_accuracy, _ = evaluate_model(best_rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "Tuned Random Forest")



# Compare models
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Tuned Random Forest']
accuracies = [lr_accuracy, rf_accuracy, svm_accuracy, best_rf_accuracy]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Extract feature importance from Random Forest model
if hasattr(best_rf_model, 'feature_importances_'):
    feature_importance = best_rf_model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance from Random Forest Model')
    plt.tight_layout()
    plt.show()
