# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sushiendar M
RegisterNumber: 212223040217 
*/
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```
## Output:

![image](https://github.com/user-attachments/assets/d5adc80c-e786-4f39-8d13-06df3ba68f2a)


![image](https://github.com/user-attachments/assets/e547bb31-178f-4d42-b7f1-9ca465f98cd5)



![image](https://github.com/user-attachments/assets/719f7af3-7db9-47cb-ab86-cca63fa66633)



![image](https://github.com/user-attachments/assets/6b14582e-bcbf-4158-a3a0-e919482046b1)



![image](https://github.com/user-attachments/assets/c83efea5-ec9a-40f9-a6bb-e9610906b44d)



![image](https://github.com/user-attachments/assets/cd56b321-ad98-4fd6-ad85-8fe8e6e682c9)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
