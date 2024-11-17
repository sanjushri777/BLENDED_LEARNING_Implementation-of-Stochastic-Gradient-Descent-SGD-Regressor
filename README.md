# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Import the required libraries.

2. **Load Dataset**:  
   Load the dataset into your environment.

3. **Data Preprocessing**:  
   Handle missing values and encode categorical variables.

4. **Define Features and Target**:  
   Separate the data into features (X) and the target variable (y).

5. **Split Data**:  
   Divide the data into training and testing sets.

6. **Initialize SGD Regressor**:  
   Create an SGD Regressor model.

7. **Train the Model**:  
   Fit the model on the training dataset.

8. **Evaluate Performance**:  
   Assess the model's performance using evaluation metrics.

9. **Make Predictions & Visualize**:  
   Make predictions and visualize the results.

## Program:
```python
'''
Program to implement SGD Regressor for linear regression.
Developed by: Sanjushri A
RegisterNumber: 21223040187
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\admin\Downloads\X_Y_Sinusoid_Data (2).csv'  # Replace with the actual file path
data = pd.read_csv(file_path)
print(data.columns)


# Split the data into features and target
X = data[['x']]  # Independent variable
y = data['y']    # Dependent variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SGD Regressor
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_regressor.fit(X_train, y_train)

# Predict on training and testing data
y_train_pred = sgd_regressor.predict(X_train)
y_test_pred = sgd_regressor.predict(X_test)

# Evaluate performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training Mean Squared Error:", train_mse)
print("Testing Mean Squared Error:", test_mse)
print("Training R^2 Score:", train_r2)
print("Testing R^2 Score:", test_r2)

# Visualize predictions vs actual data
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_test_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/e4de9c85-8719-4681-beab-5b940b5df623)





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
