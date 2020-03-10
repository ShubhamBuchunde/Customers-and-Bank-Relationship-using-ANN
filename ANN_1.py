# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:57:47 2020

@author: Shubham Buchunde
"""
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x_1 = LabelEncoder()
x[:,1] = label_x_1.fit_transform(x[:,1])
label_x_2 = LabelEncoder()
x[:,2] = label_x_2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2- Making of ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,kernel_initializer= 'uniform',activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units = 6,kernel_initializer= 'uniform',activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1,kernel_initializer= 'uniform',activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train,y_train,batch_size=10, epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = new_prediction>0.5

from keras.wrappers.scikit_learn  import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    
    classifier = Sequential()

    classifier.add(Dense(units = 6,kernel_initializer= 'uniform',activation = 'relu', input_dim = 11))

    classifier.add(Dense(units = 6,kernel_initializer= 'uniform',activation = 'relu'))

    classifier.add(Dense(units = 1,kernel_initializer= 'uniform',activation = 'sigmoid'))

    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier,batch_size= 10,epochs=100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = 1)
mean= accuracies.mean()
variance = accuracies.std()




























