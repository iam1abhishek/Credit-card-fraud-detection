## Credit Card Fraud Detection using Machine Learning

### Overview
This repository contains code for detecting credit card fraud using machine learning algorithms, specifically Random Forest and Decision Tree classifiers. The dataset used consists of 31 parameters, with features like time, transaction amount, and PCA-transformed variables.

### Dataset Description
The dataset comprises 284,807 transactions, out of which only 492 are fraud cases, leading to an imbalanced class distribution. Features like 'Time' and 'Amount' were not transformed using PCA.
Dataset used(Drive linke): https://drive.google.com/file/d/1xU4JazBSKRDWkeYA0_6BSH9QhLOadx1u/view?usp=sharing

### Getting Started
1. Import necessary modules for data processing and machine learning.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
```

2. Load the dataset and perform initial data exploration.
```python
dataframe = pd.read_csv("./Desktop/DataFlair/credit_card_fraud_detection/creditcard.csv")
dataframe.head()
dataframe.info()
```

3. Check for null values and visualize the class distribution.
```python
dataframe.isnull().values.any()
dataframe["Amount"].describe()
```

4. Perform data preprocessing by scaling and splitting the data.
```python
scaler = StandardScaler()
dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
dataframe.drop(["Amount", "Time"], inplace=True, axis=1)
```

5. Train different models (Random Forest and Decision Tree) on the dataset.
```python
# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_Y)
```

6. Evaluate the models and visualize performance using confusion matrices.
```python
confusion_matrix_dt = confusion_matrix(test_Y, predictions_dt.round())
confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())

# Print scores and metrics
print("Random Forest Score: ", random_forest_score)
print("Decision Tree Score: ", decision_tree_score)
metrics(test_Y, predictions_dt.round())
metrics(test_Y, predictions_rf.round())
```

7. Handle class imbalance using oversampling (SMOTE) and retrain the Random Forest classifier.
```python
from imblearn.over_sampling import SMOTE

X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)

rf_resampled = RandomForestClassifier(n_estimators=100)
rf_resampled.fit(train_X, train_Y)
```

8. Evaluate the retrained Random Forest model and visualize the improved performance.
```python
cm_resampled = confusion_matrix(test_Y, predictions_resampled.round())
print("Confusion Matrix - Random Forest")
print(cm_resampled)

metrics(test_Y, predictions_resampled.round())
```

### Conclusion
By applying machine learning techniques like Random Forests and Decision Trees, we can effectively detect fraudulent credit card transactions, even with imbalanced datasets. Oversampling techniques like SMOTE can further improve model performance.
