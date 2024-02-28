# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network attempts to solve a fundamental problem: finding the best way to transform inputs into desired outputs. It does this by learning complex patterns and relationships within data.
A neural network is a computational model inspired by the structure and function of the human brain. It is a type of machine learning algorithm that processes information through interconnected nodes, known as neurons or artificial neurons. These neurons are organized into layers: an input layer, one or more hidden layers, and an output layer. In a neural network, each connection between neurons has an associated weight, and the network learns by adjusting these weights based on input data and desired output

## Neural Network Model

![images](https://github.com/chaitanya18c/basic-nn-model/assets/119392724/4d3f9a8f-5992-4285-b4a3-520c261b03c3)

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

```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('EX1_DL').sheet1
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
  y.append(num**2)
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
history=model.fit(Input_train,Output_train, epochs=1000,batch_size=32)
loss_df=pd.DataFrame(model.history.history)
loss_df.plot()
X_test1 =Scaler.transform(Input_test)
model.evaluate(X_test1,Output_test)
X_n1= [[20]]
X_n11=Scaler.transform(X_n1)
model.predict(X_n11)
```

## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT
