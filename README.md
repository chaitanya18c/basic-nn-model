![nn](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/77c7258d-a9a2-4a14-ad6c-db5a4bec7bc1)# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network attempts to solve a fundamental problem: finding the best way to transform inputs into desired outputs. It does this by learning complex patterns and relationships within data.
A neural network is a computational model inspired by the structure and function of the human brain. It is a type of machine learning algorithm that processes information through interconnected nodes, known as neurons or artificial neurons. These neurons are organized into layers: an input layer, one or more hidden layers, and an output layer. In a neural network, each connection between neurons has an associated weight, and the network learns by adjusting these weights based on input data and desired output

## Neural Network Model
![image](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/3530847c-86d8-4d4e-86e1-9c38140de56f)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
# Name : CHAITANYA P S
# Register Number : 212222230024
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DLdata1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
x=[]
y=[]
for i in range(60):
  num = i+1
  x.append(num)
  y.append(num*2)
df=pd.DataFrame({'INPUT': x, 'OUTPUT': y})
df.head()
inp=df[["INPUT"]].values
out=df[["OUTPUT"]].values
Input_train,Input_test,Output_train,Output_test=train_test_split(inp,out,test_size=0.33)
Scaler=MinMaxScaler()
Scaler.fit(Input_train)
Input_trains=Scaler.transform(Input_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential([Dense(5,activation='relu'),
                  Dense(10,activation='relu'),
                  Dense(1)])
model.compile(loss="mse",optimizer="rmsprop")
history=model.fit(Input_train,Output_train, epochs=3500)
loss_df=pd.DataFrame(model.history.history)
loss_df.plot()
X_test1 =Scaler.transform(Input_test)
model.evaluate(X_test1,Output_test)
X_n1= [[21]]
X_n11=Scaler.transform(X_n1)
model.predict(X_n11)
```

## Dataset Information
![image](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/802b974e-cb65-47ba-b141-70e94641c2a8)

## OUTPUT
### Training Loss Vs Iteration Plot
![image](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/6144ab3d-ff10-4c16-960c-6cffa3cb1023)

### Test Data Root Mean Squared Error
![image](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/5839b6dc-e0b5-4ca3-8869-91278c79f24a)

### New Sample Data Prediction
![image](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/f7666cc7-00c0-4654-9f8c-12ec7aababd4)

## RESULT
Thus the Process of developing a neural network regression model for the created dataset is successfully executed.
