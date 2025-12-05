# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset containing house features (area, rooms, age) and targets (price, occupants).
2. Split the data and train a MultiOutputRegressor with SGDRegressor using the training set.
3. Input new house details (area, rooms, age) from the user.
4. Predict and display the house price and number of occupants using the trained model.

## Program:
```
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# Sample dataset
# Features: [Area (sqft), Num of Rooms, Age of House]
X = np.array([
    [1200, 3, 10],
    [1500, 4, 5],
    [800, 2, 20],
    [2000, 5, 2],
    [950, 2, 15],
    [1750, 4, 8]
])

# Targets: [Price, Occupants]
y = np.array([
    [50, 4],
    [70, 5],
    [30, 3],
    [90, 6],
    [35, 3],
    [80, 5]
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SGD Regressor (wrapped for multi-output)
model = MultiOutputRegressor(SGDRegressor(max_iter=2000, tol=1e-3))
model.fit(X_train, y_train)

# Take user input
area = float(input("Enter house area (sqft): "))
rooms = int(input("Enter number of rooms: "))
age = float(input("Enter age of the house: "))

# Predict
prediction = model.predict([[area, rooms, age]])[0]

print("Predicted House Price (in lakhs):", round(prediction[0], 2))
print("Predicted Number of Occupants:", round(prediction[1]))

/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Shafi Ahmed MS
RegisterNumber: 25014933
*/
```

## Output:

<img width="616" height="149" alt="image" src="https://github.com/user-attachments/assets/31ba28dc-2b9a-4216-9e43-0ddc70ae3472" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
