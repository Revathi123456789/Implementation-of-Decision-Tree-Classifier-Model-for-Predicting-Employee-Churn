# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Ayshwariya J

RegisterNumber: 212224230030
```python
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![Image-1](https://github.com/user-attachments/assets/63d8de02-c292-4ae0-ae86-0048664fbabb)

![Image-2](https://github.com/user-attachments/assets/6df4f2cb-692e-4728-9157-b745c3970dd6)

![Image-3](https://github.com/user-attachments/assets/d45b4086-005f-4e6c-90b2-ecff57cbebdc)

![Image-4](https://github.com/user-attachments/assets/0bd4e6a5-c748-42ea-9cd5-afe8c20a3857)

![Image-5](https://github.com/user-attachments/assets/06ea0246-1bee-43f8-a96b-5dd8bdd032de)

![Image-6](https://github.com/user-attachments/assets/45c2ff77-9abf-4bb5-ba9d-4a3558f3d6b2)

![Image-7](https://github.com/user-attachments/assets/5c9c98be-e26c-41af-9a1f-3c5c7e166659)

![Image-8](https://github.com/user-attachments/assets/99637a0f-92f9-435a-9e00-eb680cc17741)










## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
