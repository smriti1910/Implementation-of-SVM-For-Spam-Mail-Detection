# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect the encoding of the spam.csv file.
2. Load the spam.csv file into a Pandas DataFrame.
3. Split the data into training and test sets. 
4. Create a bag-of-words representation of the data.
5. Train a support vector machine (SVM) model.
6. Predict the labels for the test data.
7. Evaluate the performance of the model. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SMRITI .B 
RegisterNumber:  212221040156
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)\


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## 1.Result
![image](https://github.com/smriti1910/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133334803/a4473fa0-2e99-4ded-bd4f-62cbbd64f2c2)
## 2.data.head()
![image](https://github.com/smriti1910/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133334803/43914470-9267-4491-a533-eeb82cdeb606)
## 3.data.info()
![image](https://github.com/smriti1910/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133334803/47ff2c47-190a-41b8-b5a9-26ca8b01492a)
## 4. data.isnull().sum()
![image](https://github.com/smriti1910/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133334803/14546688-5ee8-47a3-b6a9-62378dca06d1)
## 5.  Y_prediction value
![image](https://github.com/smriti1910/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133334803/71250e21-ea8f-445f-a815-9d17bfd2a659)
## 6. Accuracy value
![image](https://github.com/smriti1910/Implementation-of-SVM-For-Spam-Mail-Detection/assets/133334803/2c252583-8a00-444e-80f5-b171f3cc2e44)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
